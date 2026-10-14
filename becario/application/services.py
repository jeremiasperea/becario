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
import time
from dataclasses import replace
from functools import partial
from typing import Callable, Optional

from pydantic import ValidationError

from ..incidentes import FALLO_TRAS_CONFIRMAR, nuevo_incidente
from ..domain.fuera_de_alcance import fuera_de_alcance
from ..domain.models import (
    Intent,
    descartar_numeros_inventados,
    PendingAction,
    PendingEdit,
    PendingPlan,
    Plan,
    PlanStep,
    RouterUnavailableError,
)
from ..domain.ports import (
    CalcInputGenerator,
    CalcRunRepository,
    ClusterGatewayFactory,
    ConfirmationStore,
    HistoryRepository,
    IntentRouter,
    JobTracker,
    PendingEditStore,
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
from . import plan_coordinator, plan_describer
from .rechazos import registrar_rechazo, PRE_ROUTER, ROUTER
# Compatibilidad: los textos se mudaron a `textos.py`; se re-exportan acá
# porque tests y presentación los importan desde `services`.
from .textos import (  # noqa: F401
    ALREADY_MODIFYING_TEXT,
    EXPIRED_CONFIRMATION_TEXT,
    FOREIGN_CONFIRMATION_TEXT,
    HELP_TEXT,
    NOT_REGISTERED_TEXT,
    ROUTER_ERROR_TEXT,
    ROUTER_TIMEOUT_TEXT,
    ROUTER_UNREACHABLE_TEXT,
    USED_CONFIRMATION_TEXT,
    _router_failure_text,
)
# Compatibilidad: el store en memoria y el reconocimiento de «cancelar» se
# mudaron a `pending_edits.py`.
from .pending_edits import (  # noqa: F401
    _CANCEL_RE,
    _InMemoryPendingEdits,
    _is_cancel_message,
)

logger = logging.getLogger(__name__)

# `_PendingEdit` se mudó al dominio como `PendingEdit`: lo necesita también
# la infraestructura para poder guardarlo (`SQLitePendingEditStore`). El
# alias mantiene los nombres internos de este módulo sin tocar 20 firmas.
_PendingEdit = PendingEdit


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
        pending_edits: Optional[PendingEditStore] = None,
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
        # Pedidos esperando respuesta, por usuario. El store se inyecta:
        # producción usa el de SQLite (un reinicio se llevaba la repregunta
        # en silencio y el usuario recibía "no pude interpretar tu pedido"
        # por una respuesta perfecta), y el default en memoria mantiene los
        # tests sin base.
        self._edit_ttl = edit_ttl_seconds
        self._pending_edits: PendingEditStore = (
            pending_edits if pending_edits is not None else _InMemoryPendingEdits()
        )

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

        # ¿Pide algo que directamente no se sabe hacer? Va ANTES del router
        # y no se registra como decisión: el router nunca se pronunció, y
        # meter esto en el dataset lo ensuciaría con mensajes que no ruteó
        # —misma razón por la que tampoco se registra un router caído—.
        if (limite := fuera_de_alcance(text)) is not None:
            registrar_rechazo(PRE_ROUTER, user_id=user_id, detalle=text)
            return Reply(text=limite, ok=False)

        started = time.monotonic()
        try:
            plan = self._route_message(text)
        except RouterUnavailableError as exc:
            # El modelo no llegó a pronunciarse: NO se cae al camino de
            # `UNKNOWN`, que le echaría la culpa al mensaje del usuario. Y
            # no se registra la decisión: no hubo decisión que registrar, y
            # meterla como 'error' ensuciaría el dataset del router con
            # fallos que no son suyos.
            logger.error(
                "Router no disponible (%s) para user=%s: %s",
                exc.reason.value, user_id, exc,
            )
            self._registrar_fallo_de_router(exc)
            return Reply(text=_router_failure_text(exc), ok=False)
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
        #
        # `awaiting_params` queda afuera a propósito: una repregunta (falta
        # la fase de un compuesto, el índice de Miller de una losa) es un
        # pedido bien ruteado al que le falta un dato, no un error de
        # ejecución — marcarla 'error' ensuciaba el dataset con decisiones
        # que el router entendió perfectamente.
        if decision_id is not None and not reply.ok and not reply.awaiting_params:
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
        """Ver `plan_describer.looks_compound`."""
        return plan_describer.looks_compound(text, cls._COMPOUND_MARKERS)

    _needs_decomposition = staticmethod(plan_describer.needs_decomposition)

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
    def _registrar_fallo_de_router(self, exc: RouterUnavailableError) -> None:
        """Deja constancia de que el modelo no contestó.

        No propaga: el usuario ya tiene su mensaje —el honesto, el que dice
        que expiró y no que no se lo entendió— y perderlo porque falló la
        contabilidad sería cambiar lo importante por lo accesorio.
        """
        if self._decision_log is None:
            return
        try:
            self._decision_log.add_failure(exc.reason.value)
        except Exception:
            logger.exception("no se pudo registrar el fallo del router")

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
                # El router se pronunció y no hay handler para lo que dijo
                # (`UNKNOWN`, o una acción sin handler). Es un rechazo, no un
                # fallo: el texto es el mismo de siempre, pero ahora deja
                # rastro de CUÁL de los tres portones se cerró.
                registrar_rechazo(
                    ROUTER, user_id=ctx.user_id, detalle=step.action.value
                )
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
        """Ver `plan_describer.is_batch_plan`."""
        return plan_describer.is_batch_plan(plan, cls._MATERIALIZABLE_STEP_INTENTS)

    _describe_structure_step = staticmethod(plan_describer.describe_structure_step)

    @classmethod
    def _describe_batch_step(cls, step: PlanStep) -> str:
        """Ver `plan_describer.describe_batch_step`."""
        return plan_describer.describe_batch_step(step, cls._describe_structure_step)

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
                # La ambigüedad de fase de Materials Project (¿tetragonal o
                # monoclínica?) es el mismo caso que la cara de la losa: si
                # se descubre recién al EJECUTAR el batch ya confirmado, la
                # repregunta pierde el `awaiting_params` en el camino
                # (`execute_calc`) y no queda nada esperando la respuesta.
                # Se adelanta acá, antes de stagear — la consulta a MP es
                # de solo lectura, no toca el cluster.
                pregunta_fase = calc.phase_question(self, req)
                if pregunta_fase is not None:
                    self._arm_if_awaiting(ctx, plan, i, pregunta_fase)
                    return pregunta_fase
                n_calc += 1
                line = calc.describe_calc_request(req)
            else:
                # Los pasos de estructura también se validan ANTES de
                # stagear: un slab sin cara no se puede armar, y stagearlo
                # haría que el usuario apruebe un plan incumplible y se
                # entere recién al confirmar. Mismo trato que el cálculo.
                if step.action is Intent.MODIFY_STRUCTURE:
                    problema = calc.validate_structure_params(step.parametros)
                    if problema is not None:
                        self._arm_if_awaiting(ctx, plan, i, problema)
                        return problema
                line = self._describe_batch_step(step)
            preview.append(f"{i}. {line}")
            actions.append(PendingAction(
                chat_id=ctx.chat_id, requester_id=ctx.user_id,
                intent=step.action, description=line, payload=dict(step.parametros),
                # El pedido original viaja con cada paso, igual que en el
                # camino de un solo cálculo: es lo que habilita el ✏️. Sin
                # esto, frente al batch MÁS equivocado de la sesión —ocho
                # pasos, con el material y la red mal— las únicas salidas
                # eran ✅ y ❌: aprobar algo que no es lo pedido, o tirar el
                # plan entero y volver a escribirlo.
                request_intent=step.action, request_params=dict(step.parametros),
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
            allow_modify=pending.allow_modify,
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
            Intent.EXPLAIN: partial(queries.explain, self),
            Intent.SUGGEST: partial(queries.suggest, self),
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
        """Ver `plan_coordinator.materialize_step_fn`."""
        return plan_coordinator.materialize_step_fn(self, ctx, step, index, awaiting_out)

    def _run_composite_plan(self, ctx: _Ctx, plan: Plan) -> Reply:
        """Ver `plan_coordinator.run_composite_plan`."""
        return plan_coordinator.run_composite_plan(self, ctx, plan)

    def _prepare_destructive_tail(
        self, ctx: _Ctx, step: PlanStep, prefix_steps: list[PlanStep], prefix_report: list[str],
    ) -> Reply:
        """Ver `plan_coordinator.prepare_destructive_tail`."""
        return plan_coordinator.prepare_destructive_tail(
            self, ctx, step, prefix_steps, prefix_report
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
        edit = self._pending_edits.pop(user_id)
        if edit is None:
            return None, None
        if edit.expired(self._edit_ttl):
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
        if self._pending_edits.has(requester_id):
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
        vencidos = self._pending_edits.pop_expired(self._edit_ttl)
        avisos = []
        for user_id, edit in vencidos:
            logger.info("Pedido pendiente vencido para user=%s", user_id)
            if edit.chat_id:
                avisos.append((edit.chat_id, self._expired_pending_text(edit)))
        self._purgar_confirmaciones_vencidas()
        return avisos

    def _purgar_confirmaciones_vencidas(self) -> None:
        """Saca de la base las confirmaciones que nadie tocó a tiempo.

        `purge_expired()` estaba en el puerto y en las dos implementaciones
        desde que las confirmaciones se persistieron, y **no lo llamaba
        nadie**: una fila por cada tarjeta que alguien dejó vencer, para
        siempre. Nada se rompía —`peek` y `pop` miran el TTL, así que una
        vencida nunca se ejecuta— pero la tabla solo crecía.

        Va acá y no en un barrido aparte porque el vencimiento de una
        confirmación es exactamente lo mismo que ya se está atendiendo dos
        líneas más arriba, con el mismo reloj y en el mismo tick.

        Nunca interrumpe: si la base falla, el aviso de los pendientes —que
        es lo que el usuario está esperando— tiene que salir igual.
        """
        try:
            cuantas = self._confirmations.purge_expired()
        except Exception:
            logger.exception("No pude purgar las confirmaciones vencidas")
            return
        if cuantas:
            logger.info("Confirmaciones vencidas purgadas: %s", cuantas)

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
        self._pending_edits.put(
            user_id,
            PendingEdit(
                steps=steps, awaiting_index=awaiting_index, chat_id=chat_id
            ),
        )

    _plan_steps_as_requests = staticmethod(plan_describer.plan_steps_as_requests)
    _propagate_answer = staticmethod(plan_describer.propagate_answer)

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

    _render_plan_context = staticmethod(plan_describer.render_plan_context)

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
            try:
                new_params = self._router.extract_params(text)
            except RouterUnavailableError as exc:
                return self._router_failure_con_plan_vivo(ctx, edit, exc)
            new_params, inventados = descartar_numeros_inventados(new_params, text)
            if inventados:
                logger.warning(
                    "descartados por número inventado (user=%s): %s",
                    ctx.user_id, inventados,
                )
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
            elif not reply.ok:
                # El cambio no se pudo aplicar (parámetros inválidos, un
                # rechazo del dominio…). El plan de ANTES seguía siendo
                # válido, así que vuelve al estante para reintentar: el
                # docstring de arriba promete que solo un «cancelar» lo tira,
                # y sin esta rama un error de validación se lo llevaba puesto.
                self._arm_pending_edit(ctx.user_id, ctx.chat_id, edit.steps)
            return reply

        if edit.awaiting_index is not None:
            # Sabemos QUÉ paso está incompleto, así que no hay nada que
            # targetear: se usa el extractor simple y el dato va derecho al
            # hueco. Targetear "la (001)" sería inventarse un problema que
            # ya está resuelto — y saldría ambiguo casi siempre.
            return self._fill_awaiting_step(ctx, edit, text)

        plan_context = self._render_plan_context(edit.steps)
        try:
            target_index, delta = self._router.extract_edit(plan_context, text)
        except RouterUnavailableError as exc:
            return self._router_failure_con_plan_vivo(ctx, edit, exc)
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

    def _router_failure_con_plan_vivo(
        self,
        ctx: _Ctx,
        edit: _PendingEdit,
        exc: RouterUnavailableError,
        *,
        awaiting_index: Optional[int] = None,
    ) -> Reply:
        """Fallo de infraestructura mientras se editaba/completaba un plan.

        Lo importante acá no es el mensaje: es que el plan vuelve al estante
        con el TTL renovado. Un fallo del servidor no puede costarle al
        usuario un pedido que ya estaba armado — el docstring de
        `_apply_edit` promete que solo un «cancelar» explícito lo descarta, y
        esa promesa vale también cuando quien falla somos nosotros.
        """
        logger.error(
            "Router no disponible (%s) editando el plan de user=%s: %s",
            exc.reason.value, ctx.user_id, exc,
        )
        self._arm_pending_edit(
            ctx.user_id, ctx.chat_id, edit.steps, awaiting_index=awaiting_index
        )
        return Reply(
            text=_router_failure_text(exc)
            + "\n\n📌 Tu plan sigue esperando: cuando vuelva a andar, repetime "
            "el cambio y sigo desde donde estábamos.",
            ok=False,
            awaiting_params=True,
        )

    def _fill_awaiting_step(
        self, ctx: _Ctx, edit: _PendingEdit, text: str
    ) -> Reply:
        """Completa el paso que quedó esperando un dato y re-despacha el plan.

        El plan se re-arma ENTERO desde los pedidos originales: los pasos que
        se habían omitido por depender del incompleto vuelven a correr, que es
        el punto de haber guardado el plan completo y no solo el hueco."""
        assert edit.awaiting_index is not None
        try:
            new_params = self._router.extract_params(text)
        except RouterUnavailableError as exc:
            return self._router_failure_con_plan_vivo(
                ctx, edit, exc, awaiting_index=edit.awaiting_index
            )
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
        self._propagate_answer(new_steps, edit.awaiting_index, base_params, new_params)
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
        try:
            ok, text = executor(ctx, action)
        except Exception:
            return self._fallo_tras_confirmar(action)
        # `ok` se descartaba acá: un scancel/sbatch que el cluster rechazó
        # (SSH arriba, comando abajo, sin excepción) volvía como `ok=True`
        # por default — el canal de salida no tiene forma de distinguir
        # "confirmado y salió bien" de "confirmado y falló", y nada aguas
        # arriba (bitácora, tests) lo ve tampoco.
        return Reply(text=text, ok=ok)

    def _fallo_tras_confirmar(self, action: PendingAction) -> Reply:
        """Qué hacer cuando la ejecución de una acción YA confirmada revienta.

        El token se consumió más arriba y **no se devuelve**, aunque tiente:
        `pop` es atómico justamente para que dos toques de ✅ no manden dos
        `sbatch` (`storage.py`, `UPDATE ... WHERE consumed_at IS NULL`), y
        reponerlo abriría esa puerta desde el otro lado. Peor: la excepción
        pudo saltar DESPUÉS de que el `sbatch` saliera, así que reponer el
        token es ofrecerle al usuario un botón que duplica un trabajo que ya
        está en la cola.

        Lo único honesto es decir que no sabemos, y mandar a verificar en vez
        de a reintentar. El id del incidente conecta este mensaje con el
        traceback del log.
        """
        incidente = nuevo_incidente()
        logger.exception(
            "Incidente %s ejecutando %s tras confirmar (token ya consumido)",
            incidente, action.intent.value,
        )
        return Reply(text=FALLO_TRAS_CONFIRMAR.format(incidente=incidente), ok=False)

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
            try:
                reply = self._execute_batch_step(ctx, action)
            except Exception:
                # Un paso que revienta corta el batch igual que uno que
                # falla (misma semántica sin-rollback de ADR-0006): lo ya
                # ejecutado queda hecho y los que siguen se omiten. La
                # diferencia es que de ESTE paso no sabemos el desenlace.
                reply = self._fallo_tras_confirmar(action)
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
        self._descartar_pendientes(requester_id, plan)
        return Reply(text="❌ Operación cancelada.")

    def _descartar_pendientes(self, requester_id: int, plan: PendingPlan) -> None:
        """Borra del cluster los inputs de una corrida que se canceló.

        Nunca hace fallar la cancelación: cancelar es lo que el usuario
        pidió y ya está hecho: el token se consumió. Si la limpieza no sale,
        queda el barrido por edad (`sweep_pending`), que existe justamente
        porque este camino puede fallar o ni siquiera ejecutarse — nadie
        aprieta ❌ cuando deja vencer una confirmación.
        """
        pendientes = [
            paso.payload["pending_dir"]
            for paso in plan.steps
            if paso.payload.get("pending_dir")
        ]
        if not pendientes:
            return
        try:
            identity = self._registry.get_identity(requester_id)
            if identity is None:  # pragma: no cover - se dio de baja en el medio
                return
            cluster = self._cluster_factory.for_identity(identity)
            for ruta in pendientes:
                resultado = cluster.discard_pending(ruta)
                if not resultado.ok:
                    logger.warning(
                        "No pude borrar la corrida cancelada %s: %s",
                        ruta, resultado.message,
                    )
        except Exception:
            logger.exception("Falló la limpieza de una corrida cancelada")
