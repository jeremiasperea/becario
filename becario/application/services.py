"""Casos de uso de B.E.C.A.R.I.O.

Esta capa orquesta el dominio y los puertos. No conoce Telegram, paramiko
ni Ollama: recibe texto, devuelve objetos `Reply`. La presentación decide
cómo renderizarlos (mensaje simple o botones de confirmación).

Modelo multiusuario: no existe un rol admin dentro del bot. Cada mensaje
se resuelve a la `ClusterIdentity` de quien lo mandó (vía `UserRegistry`)
y las operaciones sobre el cluster corren con SU propia cuenta SSH
(vía `ClusterGatewayFactory`) — el aislamiento entre personas lo garantiza
Slurm/el sistema operativo del cluster, no lógica de la aplicación.

La lógica de cada intent vive en `handlers/` (jobs, calc, queries,
remote_files); esta fachada resuelve identidad, arma el contexto y delega.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Optional

from pydantic import ValidationError

from ..domain.models import (
    _MAX_AUTOMATERIALIZE_STEPS,
    Intent,
    PendingAction,
    PendingPlan,
    Plan,
    PlanStep,
)
from ..domain.ports import (
    CalcInputGenerator,
    CalcRunRepository,
    ClusterGatewayFactory,
    ConfirmationStore,
    HistoryRepository,
    IntentRouter,
    JobTracker,
    RouterDecisionLog,
    StructureBuilder,
    StructureProvider,
    UserRegistry,
)
from .context import Reply, _Ctx
from .handlers import calc, jobs, queries, remote_files
# Compatibilidad: estos helpers vivían acá y se importan desde afuera
# (tests, presentación); se re-exportan desde su nuevo módulo. Ojo: son
# alias — monkeypatchear estos nombres acá NO afecta a quien los usa en
# `handlers.remote_files` (parchear allá si hace falta).
from .handlers.remote_files import (  # noqa: F401
    _LISTING_MAX_CHARS,
    _VIEW_FILE_MAX_BYTES,
    _truncate_listing,
)
from .plan_executor import PlanExecutor

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "❓ No pude interpretar tu pedido. Probá con:\n"
    '• "Relajá los parámetros de red del bulk de W"\n'
    '• "Hacé la curva de convergencia de ENCUT para Zr hcp"\n'
    '• "Dame los parámetros de red del cálculo del Zr"\n'
    '• "Mostrá el estado de mis trabajos"\n'
    '• "Cancelá el trabajo 12345"\n'
    '• "Consultá el historial"\n'
    '• "Generá un POSCAR de Si diamond 2x2x2"\n'
    '• "Creame la carpeta /home/usuario/pruebas"\n'
    '• "Qué archivos hay en /data/becario_runs"'
)

NOT_REGISTERED_TEXT = (
    "🚫 No estás registrado en B.E.C.A.R.I.O. Pedile a quien administra "
    "el grupo que te agregue con tu usuario del cluster."
)

FOREIGN_CONFIRMATION_TEXT = "⚠️ Esta confirmación no te pertenece."
# Se conserva el texto histórico para el caso genérico; los casos que sí se
# pueden distinguir tienen el suyo (ver `_stale_confirmation_reply`).
EXPIRED_CONFIRMATION_TEXT = "⌛ Esta confirmación expiró o ya fue usada."
USED_CONFIRMATION_TEXT = (
    "✔️ Ese botón ya se usó — la acción se hizo con el primer toque. "
    "Si no viste la respuesta, pudo perderse en el camino: mirá el chat un "
    "momento antes de repetir el pedido."
)
ALREADY_MODIFYING_TEXT = (
    "✏️ Ya estás modificando ese plan: apretaste ✏️ y quedó esperando tu "
    "cambio. Decímelo y sigo (p. ej. «usá 2 nodos», «subí el ENCUT a 600»), "
    "o escribí «cancelar» para descartarlo."
)


@dataclass
class _PendingEdit:
    """El usuario tocó ✏️ Modificar: su próximo mensaje describe el
    cambio. Guarda el pedido ORIGINAL (acción + parámetros) de cada paso
    editable del plan, en orden — un plan de un solo paso es la lista de
    largo 1 de siempre; un plan de varios pasos generaliza a targeting
    (tarea 5.1, diseño §3.2/§4.6): el cambio aceptado SIEMPRE re-arma el
    plan entero desde estos pedidos originales, nunca parchea un payload
    ya materializado."""

    steps: list[tuple[Intent, dict]]
    created_at: float = field(default_factory=time.time)
    # A qué chat avisarle cuando venza. Sin esto el vencimiento solo se
    # podría contar cuando el usuario vuelve a escribir, que es tarde: ya
    # se quedó esperando una respuesta que nunca iba a llegar.
    chat_id: int = 0
    # Paso (1-based) al que le faltaba un dato, cuando el pendiente lo armó
    # un handler vía `Reply.awaiting_params` en vez del botón ✏️ Modificar.
    # Sabiéndolo no hace falta targetear la respuesta: ya conocemos el hueco,
    # así que se usa el extractor simple y se evita el caso ambiguo — "la
    # (001)" no dice a qué paso pertenece, pero nosotros sí lo sabemos.
    awaiting_index: Optional[int] = None


_CANCEL_RE = re.compile(r"\A\s*(cancel|descart|olvid|dejal|no,?\s*(dejalo|nada))", re.IGNORECASE)


def _is_cancel_message(text: str) -> bool:
    """¿El mensaje pide descartar el plan en edición? («cancelar»,
    «cancelalo», «descartá todo», «olvidalo»…)."""
    return bool(_CANCEL_RE.match(text or ""))


class BecarioService:
    """Fachada de casos de uso (Single Responsibility: orquestar, no ejecutar)."""

    def __init__(
        self,
        router: IntentRouter,
        registry: UserRegistry,
        cluster_factory: ClusterGatewayFactory,
        history: HistoryRepository,
        confirmations: ConfirmationStore,
        structures: StructureBuilder,
        job_tracker: JobTracker,
        calc_inputs: Optional[CalcInputGenerator] = None,
        potcar_dir: str = "",
        remote_base: str = "becario_runs",
        calc_runs: Optional[CalcRunRepository] = None,
        edit_ttl_seconds: float = 600.0,
        decision_log: Optional[RouterDecisionLog] = None,
        structure_provider: Optional[StructureProvider] = None,
        mp_api_key: str = "",
    ) -> None:
        self._router = router
        self._registry = registry
        self._cluster_factory = cluster_factory
        self._history = history
        self._confirmations = confirmations
        self._structures = structures
        self._job_tracker = job_tracker
        self._calc_inputs = calc_inputs
        self._potcar_dir = potcar_dir.rstrip("/")
        self._remote_base = remote_base.rstrip("/")
        self._calc_runs = calc_runs
        self._decision_log = decision_log
        self._structure_provider = structure_provider
        self._mp_api_key = mp_api_key
        # Modificaciones pendientes por usuario (en memoria, con TTL: si se
        # reinicia el bot simplemente se vuelve a pedir el cálculo).
        self._edit_ttl = edit_ttl_seconds
        self._pending_edits: dict[int, _PendingEdit] = {}
        self._edits_lock = threading.Lock()

    # Orden de búsqueda del pseudopotencial de cada elemento en la
    # biblioteca del cluster (variantes semi-core primero, como recomienda
    # VASP para la mayoría de los metales de transición).
    _POTCAR_VARIANTS = ("_sv", "_pv", "")

    # ------------------------------------------------------------------
    # Entrada principal: texto del usuario
    # ------------------------------------------------------------------
    def handle_text(self, chat_id: int, user_id: int, text: str) -> Reply:
        identity = self._registry.get_identity(user_id)
        if identity is None:
            logger.warning("Mensaje de usuario no registrado: telegram_user_id=%s", user_id)
            return Reply(text=NOT_REGISTERED_TEXT)

        ctx = _Ctx(
            chat_id=chat_id,
            user_id=user_id,
            identity=identity,
            cluster=self._cluster_factory.for_identity(identity),
        )

        # ¿Está describiendo un cambio a un plan que pidió modificar?
        edit, expired = self._pop_pending_edit(user_id)
        if edit is not None:
            return self._apply_edit(ctx, edit, text)

        started = time.monotonic()
        plan = self._route_message(text)
        latency = time.monotonic() - started
        logger.info(
            "routed plan steps=%s ssh_user=%s",
            [(s.action, s.parametros) for s in plan.steps], identity.ssh_user,
        )

        decision_id = self._log_decision(chat_id, user_id, text, plan, latency)
        if decision_id is not None:
            ctx = replace(ctx, decision_id=decision_id)

        reply = self._dispatch_plan(ctx, plan)
        if expired is not None and reply.text == HELP_TEXT:
            # El mensaje no se entendió Y había un pedido vencido: casi
            # seguro estaba contestando la repregunta («tetragonal»), que
            # sola no significa nada. Decir "no te entendí" es cierto pero
            # inútil — el usuario cree que se explicó mal cuando en realidad
            # nos olvidamos de lo que le habíamos preguntado. Solo se pisa
            # el HELP_TEXT: un pedido nuevo que SÍ se entiende se ejecuta
            # normalmente, sin avisos sobre lo que caducó.
            reply = Reply(text=self._expired_pending_text(expired), ok=False)
        # Un paso que falló al ejecutarse marca la decisión como 'error':
        # señal débil (pudo fallar el cluster, no el ruteo) pero separa
        # estos casos de los 'routed' limpios al armar el dataset.
        if decision_id is not None and not reply.ok:
            self._set_decision_outcome(decision_id, "error")
        return reply

    # ------------------------------------------------------------------
    # Descomposición de pedidos largos (etapa intermedia, medida antes
    # que diseñada: en mensajes largos el modelo arma bien la estructura
    # del plan pero omite formula/red_cristalina de los cálculos)
    # ------------------------------------------------------------------
    # Marcadores baratos de pedido compuesto (coordinación/secuencia).
    _COMPOUND_MARKERS = (" y ", " luego", " despué", " despue", ";")

    def _route_message(self, text: str) -> Plan:
        """Punto de entrada de ruteo. Los pedidos largos/compuestos van
        DIRECTO a la descomposición: sobre input largo el `route()` de
        schema grande hace timeout en CPU (medido: 300s vs 21s de
        `decompose()`), y el reintento post-hoc nunca llega a dispararse
        porque depende de un `route()` que ya falló. El resto sigue el
        camino corto con `_maybe_decompose` como red de seguridad.

        Un mensaje que parece compuesto NUNCA se rutea entero: si no se pudo
        descomponer con confianza (p. ej. el plan recompuesto excede el tope
        de 5 pasos de ADR-0006), se devuelve `UNKNOWN` — el despacho responde
        con la ayuda — en vez de caer al `route()` que haría timeout."""
        if self._looks_compound(text):
            return self._decompose_plan(text) or Plan(
                steps=[PlanStep(action=Intent.UNKNOWN)]
            )
        plan = self._router.route(text)
        return self._maybe_decompose(text, plan)

    @classmethod
    def _looks_compound(cls, text: str) -> bool:
        """Heurística barata para pedidos largos/compuestos. Conservadora
        donde importa: un falso positivo es inocuo — `decompose()` devuelve
        una sola instrucción y se rutea igual (ver `_decompose_plan`); solo
        un pedido realmente compuesto justifica saltear el `route()`
        directo. No pretende clasificar: solo decidir a quién preguntar
        primero."""
        words = text.split()
        if len(words) >= 16:
            return True
        lowered = f" {text.lower()} "
        markers = sum(lowered.count(m) for m in cls._COMPOUND_MARKERS)
        return markers >= 1 and len(words) >= 10

    @staticmethod
    def _needs_decomposition(plan: Plan) -> bool:
        """Firma exacta del fallo medido: plan multi-paso con algún
        cálculo sin material."""
        if len(plan.steps) < 2:
            return False
        return any(
            s.action is Intent.PREPARE_CALC and not s.parametros.get("formula")
            for s in plan.steps
        )

    def _decompose_plan(self, text: str) -> Optional[Plan]:
        """Parte el pedido en instrucciones simples auto-contenidas y rutea
        cada una por el camino corto (confiable). Devuelve el plan
        recompuesto, o `None` si no se pudo descomponer con confianza (el
        llamador decide el fallback). Un solo `parts` es válido: un mensaje
        que parecía compuesto pero era simple se rutea como un paso."""
        parts = self._router.decompose(text)
        if not parts or len(parts) > 5:
            return None
        steps = []
        for part in parts:
            steps.extend(self._router.route(part).steps)
        if not steps:
            return None
        try:
            # El cap estructural (11, ADR-0007) lo aplica `Plan`; un batch
            # más grande cae acá como plan inválido -> None (el llamador
            # responde la ayuda en vez de tocar el cluster).
            new_plan = Plan(steps=steps)
        except (ValidationError, ValueError):
            return None
        if self._needs_decomposition(new_plan):
            return None  # no mejoró: misma falla, el llamador hace fallback
        logger.info("plan descompuesto en %d instrucciones simples", len(parts))
        return new_plan

    def _maybe_decompose(self, text: str, plan: Plan) -> Plan:
        """Reintenta un plan defectuoso vía descomposición. Ante CUALQUIER
        duda devuelve el plan original — su repregunta es mejor que un plan
        inventado."""
        if not self._needs_decomposition(plan):
            return plan
        return self._decompose_plan(text) or plan

    # ------------------------------------------------------------------
    # Registro de decisiones del router (materia prima del eval set)
    # ------------------------------------------------------------------
    def _log_decision(
        self, chat_id: int, user_id: int, text: str, plan: Plan, latency: float,
    ) -> Optional[int]:
        """Registra la decisión ruteada. Nunca propaga errores: el log es
        observabilidad, no puede tirar abajo el manejo del mensaje."""
        if self._decision_log is None:
            return None
        steps_json = json.dumps(
            [{"action": s.action.value, "parametros": s.parametros} for s in plan.steps],
            ensure_ascii=False,
        )
        try:
            return self._decision_log.add(chat_id, user_id, text, steps_json, latency)
        except Exception:
            logger.exception("no se pudo registrar la decisión del router")
            return None

    def _set_decision_outcome(self, decision_id: Optional[int], outcome: str) -> None:
        if self._decision_log is None or decision_id is None:
            return
        try:
            self._decision_log.set_outcome(decision_id, outcome)
        except Exception:
            logger.exception("no se pudo marcar outcome=%s de la decisión %s", outcome, decision_id)

    def _dispatch_plan(self, ctx: _Ctx, plan: Plan) -> Reply:
        """Arma/ejecuta un `Plan` ya construido (por `route()` o por un
        ✏️ Modificar re-armado, tarea 5.1 §4.6). Un solo paso es el camino
        de hoy, sin cambios (HC1/HC2): UNA llamada al LLM (si vino de
        `route()`) y ejecuta/confirma exactamente como antes de la
        composición de planes — reusar este método desde el flujo de
        edición es lo que mantiene esa paridad byte a byte."""
        if plan.is_single:
            step = plan.single_step
            handler = self._intent_handlers().get(step.action)
            if handler is None:
                return Reply(text=HELP_TEXT)
            reply = handler(ctx, step.parametros)
            if reply.awaiting_params:
                # Falta un dato que no se adivina: el pedido queda en el
                # estante y el próximo mensaje lo completa, en vez de
                # obligar al usuario a repetirlo entero.
                self._arm_pending_edit(
                    ctx.user_id, ctx.chat_id, [(step.action, dict(step.parametros))]
                )
            return reply

        if self._is_batch_plan(plan):
            return self._prepare_batch(ctx, plan)
        return self._run_composite_plan(ctx, plan)

    @classmethod
    def _is_batch_plan(cls, plan: Plan) -> bool:
        """Un plan es BATCH (ADR-0007) si excede la composición chica que se
        auto-materializa —— más de `_MAX_AUTOMATERIALIZE_STEPS` pasos O varios
        cálculos —— y está formado SOLO por pasos materializables y
        `preparar_calculo` (sin cola realmente destructiva). Estos planes
        salen del descompositor; se confirman enteros antes de tocar el
        cluster, así que nada se auto-ejecuta y el cap deja de gatear el
        blast-radius (lo hace la confirmación)."""
        n_calc = sum(1 for s in plan.steps if s.action is Intent.PREPARE_CALC)
        if len(plan.steps) <= _MAX_AUTOMATERIALIZE_STEPS and n_calc <= 1:
            return False
        return all(
            s.action in cls._MATERIALIZABLE_STEP_INTENTS or s.action is Intent.PREPARE_CALC
            for s in plan.steps
        )

    @staticmethod
    def _describe_batch_step(step: PlanStep) -> str:
        if step.action is Intent.CREATE_DIR:
            return f"📁 crear carpeta {step.parametros.get('destino_remoto', '?')}"
        return f"• {step.action.value}"

    def _prepare_batch(self, ctx: _Ctx, plan: Plan) -> Reply:
        """Batch (ADR-0007): valida y describe cada paso SIN tocar el cluster,
        stagea el plan entero con una sola confirmación. Recién al confirmar
        se ejecuta todo. Fail-closed: un cálculo inválido aborta el batch sin
        efectos."""
        actions: list[PendingAction] = []
        preview: list[str] = []
        n_calc = 0
        for i, step in enumerate(plan.steps, start=1):
            if step.action is Intent.PREPARE_CALC:
                req = calc._build_calc_request(self, step.parametros)
                if isinstance(req, Reply):
                    # Cálculo inválido: nada se ejecuta. Si lo que falta es un
                    # dato pedible, el batch entero queda esperando en vez de
                    # obligar a rearmar los N pasos.
                    self._arm_if_awaiting(ctx, plan, i, req)
                    return req
                n_calc += 1
                line = calc.describe_calc_request(req)
            else:
                line = self._describe_batch_step(step)
            preview.append(f"{i}. {line}")
            actions.append(PendingAction(
                chat_id=ctx.chat_id, requester_id=ctx.user_id,
                intent=step.action, description=line, payload=dict(step.parametros),
            ))
        pending = PendingPlan(
            chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=actions,
            decision_id=ctx.decision_id, execute_all=True,
        )
        token = self._confirmations.put(pending)
        header = (
            f"📋 Batch de {len(plan.steps)} pasos"
            + (f" · {n_calc} trabajo(s) SLURM al confirmar" if n_calc else "")
            + ":"
        )
        return Reply(
            text=(
                f"{header}\n" + "\n".join(preview)
                + "\n\n⚠️ Nada se ejecutó todavía. ¿Confirmás el batch entero?"
            ),
            needs_confirmation=True,
            confirmation_token=token,
        )

    def _intent_handlers(self) -> dict:
        return {
            Intent.SUBMIT_SLURM: partial(jobs.prepare_submit, self),
            Intent.CANCEL_JOB: partial(jobs.prepare_cancel, self),
            Intent.CHECK_STATUS: partial(jobs.check_status, self),
            Intent.QUERY_DB: partial(queries.query_history, self),
            Intent.MODIFY_STRUCTURE: partial(calc.modify_structure, self),
            Intent.PREPARE_CALC: partial(calc.prepare_calc, self),
            Intent.QUERY_RESULTS: partial(queries.query_results, self),
            Intent.CREATE_DIR: partial(remote_files.create_directory, self),
            Intent.LIST_FILES: partial(remote_files.list_files, self),
            Intent.VIEW_FILE: partial(remote_files.view_file, self),
        }

    # ------------------------------------------------------------------
    # Composición multi-paso (planes de más de un paso, ver ADR-0006)
    # ------------------------------------------------------------------

    # Intenciones que se pueden combinar como pasos NO finales de un plan:
    # materializan de inmediato (sin confirmación), igual que ejecutan hoy
    # solas. `enviar_slurm`/`cancelar_calculo` solo se admiten como cola
    # final (validado por `Plan._v_destructive_last`); `preparar_calculo`
    # arma su propia confirmación (deja un envío pendiente) y todavía no
    # se soporta combinado en un plan — necesitaría su propio manejo de
    # cola destructiva "virtual" (ver deviations de la tarea 4.3).
    _MATERIALIZABLE_STEP_INTENTS = frozenset({
        Intent.CHECK_STATUS,
        Intent.QUERY_DB,
        Intent.QUERY_RESULTS,
        Intent.MODIFY_STRUCTURE,
        Intent.CREATE_DIR,
        Intent.LIST_FILES,
        Intent.VIEW_FILE,
    })

    def _materialize_step_fn(
        self,
        ctx: _Ctx,
        step: PlanStep,
        index: int = 1,
        awaiting_out: Optional[list[int]] = None,
    ) -> Callable[[], tuple[bool, str]]:
        """Envuelve un paso no destructivo en el `StepFn` que espera
        `PlanExecutor`: sin argumentos, devuelve `(ok, línea_de_reporte)`.

        `awaiting_out` es la vía por la que sale el `awaiting_params` del
        handler: el contrato de `StepFn` es `(ok, texto)` y no puede llevarlo,
        así que el paso anota su índice ahí. Sin esto, un pedido de dato
        faltante dentro de un plan multi-paso se ve como un fallo cualquiera
        y el pedido se pierde."""
        handler = self._intent_handlers().get(step.action)

        def _run() -> tuple[bool, str]:
            if handler is None or step.action not in self._MATERIALIZABLE_STEP_INTENTS:
                return False, f"el paso '{step.action.value}' no se puede combinar en un plan"
            reply = handler(ctx, step.parametros)
            if reply.awaiting_params and awaiting_out is not None:
                awaiting_out.append(index)
            return reply.ok, reply.text

        return _run

    def _run_composite_plan(self, ctx: _Ctx, plan: Plan) -> Reply:
        """Un plan de más de un paso: los pasos no destructivos materializan
        EN ORDEN al construir el plan (stop-on-failure); si el plan termina
        en un paso destructivo (`enviar_slurm`/`cancelar_calculo`, único
        posible por `Plan._v_destructive_last`), la confirmación se pide
        SOLO para esa cola — el resto ya corrió y se muestra ejecutado."""
        # Cola de cálculos: pasos `preparar_calculo` contiguos al FINAL del
        # plan. Cada uno arma su propia confirmación individual (decisión
        # de producto: un botón por cálculo, nunca N envíos con un botón),
        # así que acá no hay cola destructiva batch — el prefijo
        # materializa y cada cálculo sale como followup con su token.
        n_calc_tail = 0
        for s in reversed(plan.steps):
            if s.action is Intent.PREPARE_CALC:
                n_calc_tail += 1
            else:
                break
        if n_calc_tail:
            prefix = plan.steps[:-n_calc_tail]
            if any(s.action not in self._MATERIALIZABLE_STEP_INTENTS for s in prefix):
                return Reply(text=HELP_TEXT)
            awaiting: list[int] = []
            result = PlanExecutor().run([
                self._materialize_step_fn(ctx, s, i, awaiting)
                for i, s in enumerate(prefix, start=1)
            ])
            if prefix and not result.ok:
                lines = result.report_lines + [
                    f"⏸ {n_calc_tail} cálculo(s) omitido(s): falló un paso previo."
                ]
                if awaiting:
                    # No es un fallo: falta un dato. El plan ENTERO queda
                    # esperando, así que el cálculo omitido corre en cuanto
                    # la estructura se pueda armar.
                    self._arm_pending_edit(
                        ctx.user_id, ctx.chat_id, self._plan_steps_as_requests(plan),
                        awaiting_index=awaiting[0],
                    )
                    return Reply(
                        text="\n".join(lines), ok=False, awaiting_params=True
                    )
                return Reply(text="\n".join(lines), ok=False)
            handler = self._intent_handlers()[Intent.PREPARE_CALC]
            followups = tuple(
                handler(ctx, s.parametros) for s in plan.steps[-n_calc_tail:]
            )
            for offset, followup in enumerate(followups):
                if followup.awaiting_params:
                    self._arm_if_awaiting(
                        ctx, plan, len(prefix) + offset + 1, followup
                    )
                    break
            lines = result.report_lines + [
                f"⏳ Te paso {n_calc_tail} cálculo(s), confirmá cada uno:"
            ] if prefix else [f"⏳ Te paso {n_calc_tail} cálculo(s), confirmá cada uno:"]
            return Reply(text="\n".join(lines), followups=followups)

        tail = plan.steps[-1]
        has_destructive_tail = tail.action in Intent.destructive()
        prefix = plan.steps[:-1] if has_destructive_tail else plan.steps

        # `preparar_calculo` nunca es materializable (deja su propio pedido
        # pendiente) — si aparece en `prefix` esto también lo atrapa,
        # incluso sin cola destructiva (ej. `[preparar_calculo, crear_dir]`,
        # donde `tail` termina siendo el propio `preparar_calculo`).
        unsupported = [s for s in prefix if s.action not in self._MATERIALIZABLE_STEP_INTENTS]
        if unsupported:
            # v1 no combina preparar_calculo (ni ningún otro paso "de
            # preparación") dentro de un plan multi-paso — fail-closed,
            # sin ejecutar nada (ver deviations, tarea 4.3).
            return Reply(text=HELP_TEXT)

        awaiting: list[int] = []
        step_fns = [
            self._materialize_step_fn(ctx, s, i, awaiting)
            for i, s in enumerate(prefix, start=1)
        ]
        result = PlanExecutor().run(step_fns)
        if awaiting:
            self._arm_pending_edit(
                ctx.user_id, ctx.chat_id, self._plan_steps_as_requests(plan),
                awaiting_index=awaiting[0],
            )

        if not has_destructive_tail:
            return Reply(text=result.report, awaiting_params=bool(awaiting))

        if not result.ok:
            lines = result.report_lines + [f"{len(plan.steps)}. ⏸ omitido"]
            return Reply(text="\n".join(lines), awaiting_params=bool(awaiting))

        return self._prepare_destructive_tail(ctx, tail, prefix, result.report_lines)

    def _prepare_destructive_tail(
        self, ctx: _Ctx, step: PlanStep, prefix_steps: list[PlanStep], prefix_report: list[str],
    ) -> Reply:
        """Arma la confirmación de la cola destructiva final, después de
        que el resto del plan ya materializó sin errores."""
        builder = {
            Intent.SUBMIT_SLURM: jobs.build_submit_action,
            Intent.CANCEL_JOB: jobs.build_cancel_action,
        }[step.action]
        built = builder(self, ctx, step.parametros)
        n = len(prefix_report) + 1
        if isinstance(built, Reply):
            lines = prefix_report + [f"{n}. ❌ {built.text}"]
            return Reply(text="\n".join(lines))

        # `request_intent`/`request_params` conservan el pedido original de
        # CADA paso (no solo el destructivo): así ✏️ Modificar puede
        # apuntar a un paso ya materializado (tarea 5.1) y, al aceptarse
        # el cambio, re-armar el plan entero desde estos originales.
        prefix_actions = [
            PendingAction(
                chat_id=ctx.chat_id, requester_id=ctx.user_id,
                intent=s.action, description=line, payload={},
                request_intent=s.action, request_params=dict(s.parametros),
            )
            for s, line in zip(prefix_steps, prefix_report)
        ]
        plan = PendingPlan(
            chat_id=ctx.chat_id, requester_id=ctx.user_id,
            steps=[*prefix_actions, built],
            decision_id=ctx.decision_id,
        )
        token = self._confirmations.put(plan)
        # ADR-0003: se confirma exactamente lo que se ejecuta — el detalle
        # completo del paso destructivo va en el texto, no solo el rótulo.
        head, *detail = built.description.splitlines()
        lines = prefix_report + [f"{n}. ⏳ {head} — confirmá para ejecutar"]
        lines += [f"    {d}" for d in detail]
        return Reply(
            text="\n".join(lines),
            needs_confirmation=True,
            confirmation_token=token,
            allow_modify=bool(built.request_intent),
        )

    # ------------------------------------------------------------------
    # Modificación de un plan pendiente
    # ------------------------------------------------------------------
    def _pop_pending_edit(
        self, user_id: int
    ) -> tuple[Optional[_PendingEdit], Optional[_PendingEdit]]:
        """`(vigente, vencido)`: a lo sumo uno de los dos.

        Se devuelven separados porque "no había nada esperando" y "lo que
        esperaba se venció" son cosas distintas para quien contesta. Con un
        solo `None` para ambas, un mensaje que respondía una repregunta
        —«tetragonal»— se ruteaba en frío y volvía como "no pude interpretar
        tu pedido": el usuario cree que no lo entendieron cuando en realidad
        se olvidaron de lo que había preguntado."""
        with self._edits_lock:
            edit = self._pending_edits.pop(user_id, None)
        if edit is None:
            return None, None
        if (time.time() - edit.created_at) > self._edit_ttl:
            return None, edit
        return edit, None

    def _stale_confirmation_reply(self, token: str, requester_id: int) -> Reply:
        """Qué contestar cuando el token ya no sirve. Tres casos, no uno.

        El que importa es el tercero: apretar ✏️ consume el token Y deja el
        plan esperando el cambio, así que si la respuesta se pierde en la red
        —pasó: un `ConnectTimeout` al contestarle a Telegram— el segundo
        toque encontraba el token consumido y decía "expiró". El sistema
        estaba en el estado CORRECTO, esperando el cambio, y el mensaje decía
        lo contrario: quien lo leía abandonaba un pedido que seguía vivo.
        """
        with self._edits_lock:
            editando = requester_id in self._pending_edits
        if editando:
            return Reply(text=ALREADY_MODIFYING_TEXT, awaiting_params=True)
        estado = self._confirmations.status(token)
        if estado == "consumido":
            return Reply(text=USED_CONFIRMATION_TEXT)
        return Reply(text=EXPIRED_CONFIRMATION_TEXT)

    def _minutos_ttl(self) -> int:
        return max(1, round(self._edit_ttl / 60))

    def _expired_pending_text(self, edit: _PendingEdit) -> str:
        """Qué se venció y qué había pedido, para que se pueda retomar.

        Se repite el plan anotado a propósito: quien contestó «tetragonal»
        20 minutos después ya no se acuerda de con qué venía."""
        return (
            f"⌛ Cerré la consulta anterior: no recibí respuesta en "
            f"{self._minutos_ttl()} minutos.\n\n"
            f"Lo que tenía anotado era:\n{self._render_plan_context(edit.steps)}\n\n"
            "Si querés seguir, pedímelo de nuevo completo."
        )

    def sweep_expired_pendings(self) -> list[tuple[int, str]]:
        """Cierra los pedidos vencidos y devuelve `(chat_id, aviso)`.

        La llama el tick del monitor, así que el aviso sale SOLO, cuando
        vence, en vez de esperar a que el usuario escriba. Sin esto el
        silencio es ambiguo: quien preguntó algo y no contestó a tiempo no
        tiene forma de distinguir "sigue esperando" de "se olvidó".

        Solo avisa de los que tienen `chat_id`: los armados antes de que
        existiera este campo no tienen a dónde ir, y se cierran callados."""
        vencidos: list[tuple[int, _PendingEdit]] = []
        ahora = time.time()
        with self._edits_lock:
            for user_id, edit in list(self._pending_edits.items()):
                if (ahora - edit.created_at) > self._edit_ttl:
                    del self._pending_edits[user_id]
                    vencidos.append((user_id, edit))
        avisos = []
        for user_id, edit in vencidos:
            logger.info("Pedido pendiente vencido para user=%s", user_id)
            if edit.chat_id:
                avisos.append((edit.chat_id, self._expired_pending_text(edit)))
        return avisos

    def _arm_pending_edit(
        self,
        user_id: int,
        chat_id: int,
        steps: list[tuple[Intent, dict]],
        awaiting_index: Optional[int] = None,
    ) -> None:
        """Deja un pedido esperando el próximo mensaje del usuario, con el
        TTL renovado. Lo usan tanto ✏️ Modificar como los handlers que piden
        un dato faltante (`Reply.awaiting_params`).

        Pide el `chat_id` además del `user_id` porque cuando el TTL vence
        el aviso se MANDA (ver `sweep_expired_pendings`), y para eso hay
        que saber a qué chat. Van como ids sueltos y no como `_Ctx`
        porque el botón ✏️ (`start_modification`) no tiene contexto."""
        with self._edits_lock:
            self._pending_edits[user_id] = _PendingEdit(
                steps=steps, awaiting_index=awaiting_index, chat_id=chat_id
            )

    @staticmethod
    def _plan_steps_as_requests(plan: Plan) -> list[tuple[Intent, dict]]:
        """El plan como lista de pedidos originales, que es lo que guarda
        `_PendingEdit`: re-armar SIEMPRE parte de acá, nunca de un payload
        ya materializado."""
        return [(s.action, dict(s.parametros)) for s in plan.steps]

    def _arm_if_awaiting(
        self, ctx: _Ctx, plan: Plan, index: int, reply: Reply
    ) -> None:
        """Si el paso `index` (1-based) pidió un dato, deja el plan ENTERO
        esperando. Se guarda el plan completo y no solo el paso incompleto
        porque al completarlo hay que volver a despachar todo el pedido —
        el cálculo que quedó omitido tiene que correr cuando la estructura
        finalmente se pueda armar."""
        if reply.awaiting_params:
            self._arm_pending_edit(
                ctx.user_id, ctx.chat_id, self._plan_steps_as_requests(plan), awaiting_index=index
            )

    def _render_plan_context(self, steps: list[tuple[Intent, dict]]) -> str:
        """Enumeración legible del plan pendiente: se la mandamos al LLM
        de `extract_edit` como contexto para el targeting semántico, y
        se la mostramos al usuario cuando el targeting sale ambiguo."""
        lines = []
        for i, (intent, params) in enumerate(steps, start=1):
            resumen = ", ".join(f"{k}={v}" for k, v in params.items()) or "sin parámetros"
            lines.append(f"{i}. {intent.value} ({resumen})")
        return "\n".join(lines)

    def _apply_edit(self, ctx: _Ctx, edit: _PendingEdit, text: str) -> Reply:
        """El mensaje describe el cambio. El plan NUNCA se descarta solo:
        si no se entiende (o el targeting sale ambiguo) se vuelve a
        preguntar, y solo un «cancelar» explícito lo tira.

        - Un solo paso editable: camino de hoy, byte a byte (`extract_params`,
          se mezcla con el pedido original, se llama al handler directo).
        - Varios pasos: `extract_edit` targetea el paso (semántico o
          «paso N»); un cambio aceptado re-arma el plan ENTERO desde los
          pedidos originales vía `_dispatch_plan` — nunca parchea un
          payload ya materializado (diseño §4.6)."""
        if _is_cancel_message(text):
            return Reply(
                text="❌ Plan descartado. Pedime el cálculo de nuevo cuando quieras."
            )

        if len(edit.steps) == 1:
            new_params = self._router.extract_params(text)
            if not new_params:
                # Devolver el plan al estante (con TTL renovado) para reintentar.
                self._arm_pending_edit(ctx.user_id, ctx.chat_id, edit.steps)
                return Reply(
                    text="⚠️ No entendí qué querés cambiar. Decímelo de otra "
                    'forma (p. ej. «usá 2 nodos», «ENCUT máximo 600»), o escribí '
                    "«cancelar» para descartar el plan."
                )
            intent, base_params = edit.steps[0]
            merged = {**base_params, **new_params}
            logger.info(
                "plan modificado por user=%s: base=%s cambio=%s",
                ctx.user_id, base_params, new_params,
            )
            handler = self._intent_handlers()[intent]
            reply = handler(ctx, merged)
            if reply.awaiting_params:
                # Sigue faltando el dato, pero lo que SÍ trajo este mensaje ya
                # está en `merged`: se re-arma con lo acumulado, así el llenado
                # es progresivo y no se pierde lo respondido hasta acá.
                self._arm_pending_edit(ctx.user_id, ctx.chat_id, [(intent, merged)])
            return reply

        if edit.awaiting_index is not None:
            # Sabemos QUÉ paso está incompleto, así que no hay nada que
            # targetear: se usa el extractor simple y el dato va derecho al
            # hueco. Targetear "la (001)" sería inventarse un problema que
            # ya está resuelto — y saldría ambiguo casi siempre.
            return self._fill_awaiting_step(ctx, edit, text)

        plan_context = self._render_plan_context(edit.steps)
        target_index, delta = self._router.extract_edit(plan_context, text)
        n = len(edit.steps)
        if target_index is None or not (1 <= target_index <= n) or not delta:
            # Ambiguo: NUNCA se fusiona ni se ejecuta. Plan al estante, sin
            # tocar, con TTL renovado.
            self._arm_pending_edit(ctx.user_id, ctx.chat_id, edit.steps)
            return Reply(
                text="🤔 No estoy seguro a qué paso del plan te referís. "
                "Decímelo apuntando el paso (p. ej. «paso 2: usá la "
                "partición gpu»), o escribí «cancelar» para descartar el "
                f"plan.\n\nPlan actual:\n{plan_context}"
            )

        new_steps = list(edit.steps)
        intent, base_params = new_steps[target_index - 1]
        new_steps[target_index - 1] = (intent, {**base_params, **delta})
        logger.info(
            "plan modificado (paso %s) por user=%s: base=%s cambio=%s",
            target_index, ctx.user_id, base_params, delta,
        )
        plan = Plan(steps=[PlanStep(action=i, parametros=p) for i, p in new_steps])
        return self._dispatch_plan(ctx, plan)

    def _fill_awaiting_step(
        self, ctx: _Ctx, edit: _PendingEdit, text: str
    ) -> Reply:
        """Completa el paso que quedó esperando un dato y re-despacha el plan.

        El plan se re-arma ENTERO desde los pedidos originales: los pasos que
        se habían omitido por depender del incompleto vuelven a correr, que es
        el punto de haber guardado el plan completo y no solo el hueco."""
        assert edit.awaiting_index is not None
        new_params = self._router.extract_params(text)
        if not new_params:
            self._arm_pending_edit(
                ctx.user_id, ctx.chat_id, edit.steps, awaiting_index=edit.awaiting_index
            )
            return Reply(
                text="⚠️ No pude sacar el dato de ahí. Repetímelo solo "
                '(p. ej. «(001)»), o escribí «cancelar» para descartar '
                "el pedido.",
                ok=False,
                awaiting_params=True,
            )

        new_steps = list(edit.steps)
        intent, base_params = new_steps[edit.awaiting_index - 1]
        new_steps[edit.awaiting_index - 1] = (intent, {**base_params, **new_params})
        logger.info(
            "dato faltante completado (paso %s) por user=%s: %s",
            edit.awaiting_index, ctx.user_id, new_params,
        )
        plan = Plan(steps=[PlanStep(action=i, parametros=p) for i, p in new_steps])
        return self._dispatch_plan(ctx, plan)

    def start_modification(self, token: str, requester_id: int, chat_id: int) -> Reply:
        """Botón ✏️ Modificar: descarta el plan pendiente y espera que el
        próximo mensaje del usuario describa el cambio."""
        plan = self._confirmations.peek(token)
        if plan is None:
            return self._stale_confirmation_reply(token, requester_id)
        if plan.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)
        # Todo paso que conserva su pedido original (`request_intent`) es
        # editable — en un plan de un solo paso es a lo sumo uno (camino de
        # hoy); en un plan de varios, targeting multi-paso (tarea 5.1).
        steps = [
            (s.request_intent, dict(s.request_params))
            for s in plan.steps if s.request_intent is not None
        ]
        if not steps:
            return Reply(text="⚠️ Esta acción no admite modificación.")
        self._confirmations.pop(token)
        self._arm_pending_edit(requester_id, chat_id, steps)
        if len(steps) == 1:
            return Reply(
                text="✏️ Dale, decime qué querés cambiar (p. ej. «subí el ENCUT "
                "máximo a 600», «usá la partición gpu», «supercelda 2x2x2»). "
                "Las demás condiciones del plan se mantienen."
            )
        return Reply(
            text="✏️ Dale, decime qué querés cambiar. Podés decirlo tal cual "
            "(«usá la partición gpu») o apuntar el paso («paso 2: 4 nodos»). "
            f"Las demás condiciones del plan se mantienen.\n\nPlan actual:\n"
            f"{self._render_plan_context(steps)}"
        )

    # ------------------------------------------------------------------
    # Confirmaciones
    # ------------------------------------------------------------------
    def _step_executors(self):
        """Ejecutores de la COLA destructiva de un plan (a lo sumo un
        paso, siempre el último — invariante de `Plan._v_destructive_last`).
        Mirror de `_intent_handlers()`, pero para la operación irreversible
        que se dispara recién al confirmar, no al construir el plan."""
        return {
            Intent.SUBMIT_SLURM: partial(jobs.execute_submit, self),
            Intent.CANCEL_JOB: partial(jobs.execute_cancel, self),
        }

    def confirm(self, token: str, requester_id: int) -> Reply:
        plan = self._confirmations.peek(token)
        if plan is None:
            return self._stale_confirmation_reply(token, requester_id)
        if plan.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)

        identity = self._registry.get_identity(requester_id)
        if identity is None:  # se dio de baja entre el pedido y la confirmación
            self._confirmations.pop(token)
            return Reply(text=NOT_REGISTERED_TEXT)
        cluster = self._cluster_factory.for_identity(identity)

        plan = self._confirmations.pop(token)  # recién ahora se consume
        if plan is None:
            return self._stale_confirmation_reply(token, requester_id)
        # El humano avaló el plan: etiqueta el ruteo como correcto aunque
        # la ejecución posterior falle (eso es problema del cluster, no
        # del router).
        self._set_decision_outcome(plan.decision_id, "confirmed")

        # Batch (ADR-0007): nada se materializó antes, así que confirmar
        # ejecuta TODOS los pasos en orden (stop-on-failure).
        if plan.execute_all:
            return self._execute_batch(plan, requester_id, identity, cluster)

        # El paso destructivo (si existe) es siempre el último del plan
        # (`Plan._v_destructive_last`); en un plan de un solo paso es
        # indistinguible de la `PendingAction` que confirmaba antes.
        action = plan.steps[-1]
        ctx = _Ctx(chat_id=action.chat_id, user_id=requester_id, identity=identity, cluster=cluster)
        executor = self._step_executors().get(action.intent)
        if executor is None:  # pragma: no cover - defensivo
            return Reply(text="⚠️ Acción pendiente desconocida.")
        _ok, text = executor(ctx, action)
        return Reply(text=text)

    def _execute_batch(
        self, plan: PendingPlan, requester_id: int, identity, cluster,
    ) -> Reply:
        """Ejecuta un batch confirmado, EN ORDEN y con corte al primer fallo
        (misma semántica sin-rollback de ADR-0006, ahora sobre el plan
        entero). Cada paso se reporta ✅/❌/⏸."""
        ctx = _Ctx(
            chat_id=plan.steps[0].chat_id, user_id=requester_id,
            identity=identity, cluster=cluster,
        )
        lines: list[str] = []
        ok_all = True
        for i, action in enumerate(plan.steps, start=1):
            if not ok_all:
                lines.append(f"{i}. ⏸ {action.description} — omitido")
                continue
            reply = self._execute_batch_step(ctx, action)
            # El texto del handler ya trae su propio ✅/🚀/⚠️; solo lo
            # numeramos (evita duplicar el mark).
            lines.append(f"{i}. {reply.text}")
            ok_all = ok_all and reply.ok
        return Reply(text="\n".join(lines), ok=ok_all)

    def _execute_batch_step(self, ctx: _Ctx, action: PendingAction) -> Reply:
        """Ejecuta un paso de batch al confirmar: `preparar_calculo` genera,
        sube y ENVÍA de una (la confirmación del batch ya lo avaló); el resto
        usa su handler normal, que materializa en el acto."""
        if action.intent is Intent.PREPARE_CALC:
            return calc.execute_calc(self, ctx, action.payload)
        handler = self._intent_handlers().get(action.intent)
        if handler is None:  # pragma: no cover - defensivo
            return Reply(text=f"paso '{action.intent.value}' no ejecutable", ok=False)
        return handler(ctx, action.payload)

    def reject(self, token: str, requester_id: int) -> Reply:
        plan = self._confirmations.peek(token)
        if plan is None:
            return self._stale_confirmation_reply(token, requester_id)
        if plan.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)
        self._confirmations.pop(token)
        self._set_decision_outcome(plan.decision_id, "cancelled")
        return Reply(text="❌ Operación cancelada.")
