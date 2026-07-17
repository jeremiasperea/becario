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

from ..domain.models import (
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
EXPIRED_CONFIRMATION_TEXT = "⌛ Esta confirmación expiró o ya fue usada."


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
        edit = self._pop_pending_edit(user_id)
        if edit is not None:
            return self._apply_edit(ctx, edit, text)

        started = time.monotonic()
        plan = self._router.route(text)
        latency = time.monotonic() - started
        logger.info(
            "routed plan steps=%s ssh_user=%s",
            [(s.action, s.parametros) for s in plan.steps], identity.ssh_user,
        )

        decision_id = self._log_decision(chat_id, user_id, text, plan, latency)
        if decision_id is not None:
            ctx = replace(ctx, decision_id=decision_id)

        reply = self._dispatch_plan(ctx, plan)
        # Un paso que falló al ejecutarse marca la decisión como 'error':
        # señal débil (pudo fallar el cluster, no el ruteo) pero separa
        # estos casos de los 'routed' limpios al armar el dataset.
        if decision_id is not None and not reply.ok:
            self._set_decision_outcome(decision_id, "error")
        return reply

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
            return handler(ctx, step.parametros)

        return self._run_composite_plan(ctx, plan)

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

    def _materialize_step_fn(self, ctx: _Ctx, step: PlanStep) -> Callable[[], tuple[bool, str]]:
        """Envuelve un paso no destructivo en el `StepFn` que espera
        `PlanExecutor`: sin argumentos, devuelve `(ok, línea_de_reporte)`."""
        handler = self._intent_handlers().get(step.action)

        def _run() -> tuple[bool, str]:
            if handler is None or step.action not in self._MATERIALIZABLE_STEP_INTENTS:
                return False, f"el paso '{step.action.value}' no se puede combinar en un plan"
            reply = handler(ctx, step.parametros)
            return reply.ok, reply.text

        return _run

    def _run_composite_plan(self, ctx: _Ctx, plan: Plan) -> Reply:
        """Un plan de más de un paso: los pasos no destructivos materializan
        EN ORDEN al construir el plan (stop-on-failure); si el plan termina
        en un paso destructivo (`enviar_slurm`/`cancelar_calculo`, único
        posible por `Plan._v_destructive_last`), la confirmación se pide
        SOLO para esa cola — el resto ya corrió y se muestra ejecutado."""
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

        step_fns = [self._materialize_step_fn(ctx, s) for s in prefix]
        result = PlanExecutor().run(step_fns)

        if not has_destructive_tail:
            return Reply(text=result.report)

        if not result.ok:
            lines = result.report_lines + [f"{len(plan.steps)}. ⏸ omitido"]
            return Reply(text="\n".join(lines))

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
    def _pop_pending_edit(self, user_id: int) -> Optional[_PendingEdit]:
        with self._edits_lock:
            edit = self._pending_edits.pop(user_id, None)
        if edit is None or (time.time() - edit.created_at) > self._edit_ttl:
            return None
        return edit

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
                with self._edits_lock:
                    self._pending_edits[ctx.user_id] = _PendingEdit(steps=edit.steps)
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
            return handler(ctx, merged)

        plan_context = self._render_plan_context(edit.steps)
        target_index, delta = self._router.extract_edit(plan_context, text)
        n = len(edit.steps)
        if target_index is None or not (1 <= target_index <= n) or not delta:
            # Ambiguo: NUNCA se fusiona ni se ejecuta. Plan al estante, sin
            # tocar, con TTL renovado.
            with self._edits_lock:
                self._pending_edits[ctx.user_id] = _PendingEdit(steps=edit.steps)
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

    def start_modification(self, token: str, requester_id: int) -> Reply:
        """Botón ✏️ Modificar: descarta el plan pendiente y espera que el
        próximo mensaje del usuario describa el cambio."""
        plan = self._confirmations.peek(token)
        if plan is None:
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)
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
        with self._edits_lock:
            self._pending_edits[requester_id] = _PendingEdit(steps=steps)
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
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)
        if plan.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)

        identity = self._registry.get_identity(requester_id)
        if identity is None:  # se dio de baja entre el pedido y la confirmación
            self._confirmations.pop(token)
            return Reply(text=NOT_REGISTERED_TEXT)
        cluster = self._cluster_factory.for_identity(identity)

        plan = self._confirmations.pop(token)  # recién ahora se consume
        if plan is None:
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)
        # El humano avaló el plan: etiqueta el ruteo como correcto aunque
        # la ejecución posterior falle (eso es problema del cluster, no
        # del router).
        self._set_decision_outcome(plan.decision_id, "confirmed")

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

    def reject(self, token: str, requester_id: int) -> Reply:
        plan = self._confirmations.peek(token)
        if plan is None:
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)
        if plan.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)
        self._confirmations.pop(token)
        self._set_decision_outcome(plan.decision_id, "cancelled")
        return Reply(text="❌ Operación cancelada.")
