"""Casos de uso de B.E.C.A.R.I.O.

Esta capa orquesta el dominio y los puertos. No conoce Telegram, paramiko
ni Ollama: recibe texto, devuelve objetos `Reply`. La presentación decide
cómo renderizarlos (mensaje simple o botones de confirmación).

Modelo multiusuario: no existe un rol admin dentro del bot. Cada mensaje
se resuelve a la `ClusterIdentity` de quien lo mandó (vía `UserRegistry`)
y las operaciones sobre el cluster corren con SU propia cuenta SSH
(vía `ClusterGatewayFactory`) — el aislamiento entre personas lo garantiza
Slurm/el sistema operativo del cluster, no lógica de la aplicación.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import ValidationError

from ..domain.models import (
    CalcKind,
    ClusterIdentity,
    HistoryFilter,
    Intent,
    JobId,
    ListFilesRequest,
    OutputFormat,
    PendingAction,
    PendingPlan,
    Plan,
    PlanStep,
    RemoteDirRequest,
    SlurmJobRequest,
    StructureKind,
    StructureRequest,
    TrackedJob,
    VaspCalcRequest,
)
from .job_monitor import _parse_last_e0
from .plan_executor import PlanExecutor
from ..domain.ports import (
    CalcInputGenerator,
    CalcRunRepository,
    ClusterGateway,
    ClusterGatewayFactory,
    ConfirmationStore,
    HistoryRepository,
    IntentRouter,
    JobTracker,
    StructureBuilder,
    UserRegistry,
)

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


@dataclass(frozen=True)
class Reply:
    """Respuesta neutral respecto del canal de salida."""

    text: str
    needs_confirmation: bool = False
    confirmation_token: Optional[str] = None
    # Pide fuente de ancho fijo (tablas): cada canal decide cómo lograrlo.
    monospace: bool = False
    # La acción pendiente admite "modificar el plan" (tercer botón).
    allow_modify: bool = False
    # ¿Este paso se pudo completar? Solo lo usa `PlanExecutor` (vía
    # `_materialize_step`) para decidir si un plan multi-paso corta acá;
    # el canal de salida lo ignora. Default True: un paso de un solo
    # intent (el camino de hoy) nunca lo consulta.
    ok: bool = True


@dataclass
class _PendingEdit:
    """El usuario tocó ✏️ Modificar: su próximo mensaje describe el cambio,
    que se mezcla con los parámetros del pedido original."""

    intent: Intent
    base_params: dict
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class _Ctx:
    """Contexto resuelto de un pedido: quién es y con qué cuenta opera."""

    chat_id: int
    user_id: int
    identity: ClusterIdentity
    cluster: ClusterGateway


_CANCEL_RE = re.compile(r"\A\s*(cancel|descart|olvid|dejal|no,?\s*(dejalo|nada))", re.IGNORECASE)


def _is_cancel_message(text: str) -> bool:
    """¿El mensaje pide descartar el plan en edición? («cancelar»,
    «cancelalo», «descartá todo», «olvidalo»…)."""
    return bool(_CANCEL_RE.match(text or ""))


def _parse_poscar_cell(
    poscar: Optional[str],
) -> Optional[tuple[float, float, float, float, float, float]]:
    """(a, b, c, α, β, γ) de un POSCAR/CONTCAR. None si no se pudo leer."""
    if not poscar:
        return None
    lines = poscar.splitlines()
    if len(lines) < 5:
        return None
    try:
        scale = float(lines[1].split()[0])
        vectors = [
            [float(x) * scale for x in lines[i].split()[:3]] for i in (2, 3, 4)
        ]
    except (ValueError, IndexError):
        return None
    if scale <= 0:  # escala negativa = volumen objetivo; no lo generamos
        return None

    def norm(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def angle(u: list[float], v: list[float]) -> float:
        cos = sum(a * b for a, b in zip(u, v)) / (norm(u) * norm(v))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    va, vb, vc = vectors
    if min(norm(va), norm(vb), norm(vc)) == 0:
        return None
    return (
        norm(va), norm(vb), norm(vc),
        angle(vb, vc), angle(va, vc), angle(va, vb),
    )


def _calc_fingerprint(req: VaspCalcRequest) -> str:
    """JSON canónico de un pedido de cálculo: dos pedidos con la misma
    huella son EL MISMO cálculo (el rango default del barrido se resuelve
    para que 'sin rango' y 'con el rango default' den la misma huella)."""
    data = req.model_dump()
    if req.calc_kind is CalcKind.ENCUT_SCAN:
        data["encut_values"] = req.scan_values()
    return json.dumps(data, sort_keys=True, default=str)


_LISTING_MAX_CHARS = 3500  # margen bajo el límite de 4096 de Telegram


def _truncate_listing(text: str) -> str:
    """Recorta un listado largo en un borde de línea para que la respuesta
    entre en un mensaje de Telegram."""
    text = text.strip()
    if len(text) <= _LISTING_MAX_CHARS:
        return text
    cut = text.rfind("\n", 0, _LISTING_MAX_CHARS)
    if cut <= 0:
        cut = _LISTING_MAX_CHARS
    return text[:cut].rstrip() + "\n… (listado truncado)"


def _format_history_table(rows: list[dict]) -> str:
    """Tabla de ancho fijo con el historial. Pensada para mostrarse con
    fuente monoespaciada (ver `Reply.monospace`)."""
    headers = ("Fecha", "Job", "Nombre", "Estado")
    table = [
        (
            str(r.get("fecha", ""))[:16],  # sin segundos, ocupa menos
            str(r.get("job_id", "")),
            str(r.get("nombre_trabajo", "")) or "-",
            str(r.get("estado", "")),
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(headers)]

    def fila(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    separador = "-" * (sum(widths) + 2 * (len(widths) - 1))
    return "\n".join([fila(headers), separador] + [fila(t) for t in table])


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
        # Modificaciones pendientes por usuario (en memoria, con TTL: si se
        # reinicia el bot simplemente se vuelve a pedir el cálculo).
        self._edit_ttl = edit_ttl_seconds
        self._pending_edits: dict[int, _PendingEdit] = {}
        self._edits_lock = threading.Lock()

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

        plan = self._router.route(text)
        logger.info(
            "routed plan steps=%s ssh_user=%s",
            [(s.action, s.parametros) for s in plan.steps], identity.ssh_user,
        )

        if plan.is_single:
            # Camino de hoy, sin cambios (HC1/HC2): un pedido de una sola
            # acción hace UNA llamada al LLM y ejecuta/confirma exactamente
            # como antes de la composición de planes.
            step = plan.single_step
            handler = self._intent_handlers().get(step.action)
            if handler is None:
                return Reply(text=HELP_TEXT)
            return handler(ctx, step.parametros)

        return self._run_composite_plan(ctx, plan)

    def _intent_handlers(self) -> dict:
        return {
            Intent.SUBMIT_SLURM: self._prepare_submit,
            Intent.CANCEL_JOB: self._prepare_cancel,
            Intent.CHECK_STATUS: self._check_status,
            Intent.QUERY_DB: self._query_history,
            Intent.MODIFY_STRUCTURE: self._modify_structure,
            Intent.PREPARE_CALC: self._prepare_calc,
            Intent.QUERY_RESULTS: self._query_results,
            Intent.CREATE_DIR: self._create_directory,
            Intent.LIST_FILES: self._list_files,
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
            Intent.SUBMIT_SLURM: self._build_submit_action,
            Intent.CANCEL_JOB: self._build_cancel_action,
        }[step.action]
        built = builder(ctx, step.parametros)
        n = len(prefix_report) + 1
        if isinstance(built, Reply):
            lines = prefix_report + [f"{n}. ❌ {built.text}"]
            return Reply(text="\n".join(lines))

        prefix_actions = [
            PendingAction(
                chat_id=ctx.chat_id, requester_id=ctx.user_id,
                intent=s.action, description=line, payload={},
            )
            for s, line in zip(prefix_steps, prefix_report)
        ]
        plan = PendingPlan(
            chat_id=ctx.chat_id, requester_id=ctx.user_id,
            steps=[*prefix_actions, built],
        )
        token = self._confirmations.put(plan)
        label = built.description.splitlines()[0]
        lines = prefix_report + [f"{n}. ⏳ {label} — confirmá para ejecutar"]
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

    def _apply_edit(self, ctx: _Ctx, edit: _PendingEdit, text: str) -> Reply:
        """El mensaje describe el cambio: el LLM extrae solo los parámetros
        nuevos y se mezclan sobre el pedido original (lo nuevo gana). El
        plan NUNCA se descarta solo: si no se entiende el cambio se vuelve
        a preguntar, y solo un «cancelar» explícito lo tira."""
        if _is_cancel_message(text):
            return Reply(
                text="❌ Plan descartado. Pedime el cálculo de nuevo cuando quieras."
            )
        new_params = self._router.extract_params(text)
        if not new_params:
            # Devolver el plan al estante (con TTL renovado) para reintentar.
            with self._edits_lock:
                self._pending_edits[ctx.user_id] = _PendingEdit(
                    intent=edit.intent, base_params=edit.base_params
                )
            return Reply(
                text="⚠️ No entendí qué querés cambiar. Decímelo de otra "
                'forma (p. ej. «usá 2 nodos», «ENCUT máximo 600»), o escribí '
                "«cancelar» para descartar el plan."
            )
        merged = {**edit.base_params, **new_params}
        logger.info(
            "plan modificado por user=%s: base=%s cambio=%s",
            ctx.user_id, edit.base_params, new_params,
        )
        handler = self._intent_handlers()[edit.intent]
        return handler(ctx, merged)

    def start_modification(self, token: str, requester_id: int) -> Reply:
        """Botón ✏️ Modificar: descarta el plan pendiente y espera que el
        próximo mensaje del usuario describa el cambio."""
        plan = self._confirmations.peek(token)
        if plan is None:
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)
        if plan.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)
        # Hoy todo plan tiene a lo sumo un paso editable (el que conserva
        # el pedido original); la extensión a targeting multi-paso es 5.1.
        action = next((s for s in plan.steps if s.request_intent is not None), None)
        if action is None:
            return Reply(text="⚠️ Esta acción no admite modificación.")
        self._confirmations.pop(token)
        with self._edits_lock:
            self._pending_edits[requester_id] = _PendingEdit(
                intent=action.request_intent,
                base_params=dict(action.request_params),
            )
        return Reply(
            text="✏️ Dale, decime qué querés cambiar (p. ej. «subí el ENCUT "
            "máximo a 600», «usá la partición gpu», «supercelda 2x2x2»). "
            "Las demás condiciones del plan se mantienen."
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
            Intent.SUBMIT_SLURM: self._execute_submit,
            Intent.CANCEL_JOB: self._execute_cancel,
        }

    def _execute_submit(self, ctx: _Ctx, action: PendingAction) -> tuple[bool, str]:
        # `workflow` viaja junto a los campos del job pero no es parte de
        # SlurmJobRequest (Pydantic ignora las claves extra del payload).
        req = SlurmJobRequest(**action.payload)
        result = ctx.cluster.submit_job(req)
        if result.ok and result.job_id:
            self._job_tracker.track(TrackedJob(
                job_id=result.job_id,
                owner_id=ctx.user_id,
                chat_id=action.chat_id,
                ssh_user=ctx.identity.ssh_user,
                job_name=req.job_name,
                script_path=req.script_path,
                workflow=action.payload.get("workflow", ""),
            ))
            self._history.add(
                owner_id=ctx.user_id, job_id=result.job_id,
                nombre_trabajo=req.job_name, estado="enviado",
            )
            # Cálculos VASP: registrar la huella para poder avisar si
            # se vuelve a pedir lo mismo (o algo muy similar).
            if self._calc_runs is not None and action.payload.get("calc_fingerprint"):
                self._calc_runs.add(
                    owner_id=ctx.user_id,
                    job_id=result.job_id,
                    job_name=req.job_name,
                    fingerprint=action.payload["calc_fingerprint"],
                    run_dir=action.payload.get("run_dir", ""),
                )
            text = (
                f"✅ 🚀 Trabajo enviado al cluster.\n"
                f"• Job: {result.job_id}\n"
                f"• Nombre: {req.job_name}\n"
                f"• Partición: {req.partition if req.partition != 'default' else 'la default del cluster'}\n"
                f"📌 Seguimiento activado para el job {result.job_id}: te aviso cuando termine."
            )
            return True, text
        elif result.ok:
            # sbatch aceptó el trabajo pero no pudimos leer el número de job.
            text = (
                f"✅ 🚀 {result.message}\n"
                "⚠️ No pude leer el número de job, así que no voy a poder avisarte "
                "cuando termine. Consultalo con «estado de mis trabajos»."
            )
            return True, text
        return False, f"❌ 🚀 {result.message}"

    def _execute_cancel(self, ctx: _Ctx, action: PendingAction) -> tuple[bool, str]:
        jid = JobId(value=action.payload["job_id"])
        result = ctx.cluster.cancel_job(jid)
        if result.ok:
            # Ya lo sabe (lo canceló él mismo): evita el aviso duplicado
            # del monitor cuando vea el estado CANCELLED más tarde. Como el
            # monitor tampoco lo va a registrar, queda asentado acá.
            self._job_tracker.mark_notified(jid.value, ctx.user_id)
            self._history.add(
                owner_id=ctx.user_id, job_id=jid.value,
                nombre_trabajo="", estado="cancelado",
            )
        status = "✅" if result.ok else "❌"
        return result.ok, f"{status} 🛑 {result.message}"

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
        return Reply(text="❌ Operación cancelada.")

    # ------------------------------------------------------------------
    # Casos de uso destructivos: preparan, no ejecutan
    # ------------------------------------------------------------------
    def _build_submit_action(self, ctx: _Ctx, params: dict) -> PendingAction | Reply:
        """Arma el `PendingAction` de un envío Slurm, o el `Reply` de error
        si los parámetros no validan. Separado de `_prepare_submit` para
        que la cola destructiva de un plan multi-paso (`handle_text`,
        task 4.3) pueda reusar la misma validación sin volver a envolverlo
        en un `PendingPlan` de un solo paso."""
        try:
            req = SlurmJobRequest(
                job_name=params.get("nombre_trabajo") or "becario_job",
                partition=params.get("particion") or "default",
                nodes=int(params.get("nodos") or 1),
                time_limit=params.get("tiempo_limite") or "01:00:00",
                script_path=params.get("script_remoto") or "/ruta/al/calculo.sh",
            )
        except (ValidationError, ValueError) as exc:
            return Reply(text=f"⚠️ Parámetros inválidos para el envío:\n{exc}", ok=False)

        return PendingAction(
            chat_id=ctx.chat_id,
            requester_id=ctx.user_id,
            intent=Intent.SUBMIT_SLURM,
            description=f"🚀 Enviar job Slurm (cuenta {ctx.identity.ssh_user}):\n{req.describe()}",
            payload=req.model_dump(),
            request_intent=Intent.SUBMIT_SLURM,
            request_params=dict(params),
        )

    def _prepare_submit(self, ctx: _Ctx, params: dict) -> Reply:
        built = self._build_submit_action(ctx, params)
        if isinstance(built, Reply):
            return built
        plan = PendingPlan(chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=[built])
        token = self._confirmations.put(plan)
        return Reply(
            text=f"⚠️ ¿Confirmás esta acción?\n\n{built.description}",
            needs_confirmation=True,
            confirmation_token=token,
            allow_modify=True,
        )

    def _build_cancel_action(self, ctx: _Ctx, params: dict) -> PendingAction | Reply:
        """Ídem `_build_submit_action`, para cancelar_calculo."""
        raw = params.get("job_id")
        if not raw:
            return Reply(
                text="⚠️ Falta el job_id para cancelar. Decime cuál trabajo querés cancelar.",
                ok=False,
            )
        try:
            jid = JobId(value=str(raw))
        except (ValidationError, ValueError):
            return Reply(text=f"⚠️ job_id inválido: {raw!r}", ok=False)

        return PendingAction(
            chat_id=ctx.chat_id,
            requester_id=ctx.user_id,
            intent=Intent.CANCEL_JOB,
            description=f"🛑 Cancelar el trabajo {jid} (cuenta {ctx.identity.ssh_user})",
            payload={"job_id": jid.value},
        )

    def _prepare_cancel(self, ctx: _Ctx, params: dict) -> Reply:
        built = self._build_cancel_action(ctx, params)
        if isinstance(built, Reply):
            return built
        plan = PendingPlan(chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=[built])
        token = self._confirmations.put(plan)
        return Reply(
            text=f"⚠️ ¿Confirmás esta acción?\n\n{built.description}",
            needs_confirmation=True,
            confirmation_token=token,
        )

    # ------------------------------------------------------------------
    # Casos de uso de solo lectura / no destructivos: ejecutan directo
    # ------------------------------------------------------------------
    def _check_status(self, ctx: _Ctx, params: dict) -> Reply:
        jid: Optional[JobId] = None
        raw = params.get("job_id")
        if raw:
            try:
                jid = JobId(value=str(raw))
            except (ValidationError, ValueError):
                return Reply(text=f"⚠️ job_id inválido: {raw!r}", ok=False)
        result = ctx.cluster.job_status(jid)
        return Reply(
            text=f"📊 Estado trabajos ({ctx.identity.ssh_user}):\n{result.message}",
            ok=result.ok,
        )

    def _create_directory(self, ctx: _Ctx, params: dict) -> Reply:
        raw = params.get("destino_remoto")
        if not raw:
            return Reply(
                text="⚠️ Decime la ruta de la carpeta que querés crear, "
                'p. ej.: "creame la carpeta /home/usuario/pruebas".',
                ok=False,
            )
        try:
            req = RemoteDirRequest(path=str(raw))
        except (ValidationError, ValueError):
            return Reply(
                text=f"⚠️ Ruta inválida: {raw!r}. Tiene que ser una ruta "
                "absoluta (empezar con /), sin caracteres especiales ni '..'.",
                ok=False,
            )
        result = ctx.cluster.make_directory(req.path)
        if not result.ok:
            # Sin esto, un mkdir fallido (permisos, cuota) solo se ve en el chat.
            logger.warning("mkdir remoto falló para %s: %s", req.path, result.message)
        status = "✅" if result.ok else "❌"
        return Reply(text=f"{status} 📁 {result.message}", ok=result.ok)

    def _list_files(self, ctx: _Ctx, params: dict) -> Reply:
        raw = params.get("destino_remoto")
        if not raw:
            # Sin ruta: listar la base de corridas, resuelta contra el home
            # remoto si la config es relativa (mismo criterio que al subir
            # una corrida).
            base = self._remote_base
            if not base.startswith("/"):
                home = ctx.cluster.home_dir()
                if not home:
                    return Reply(
                        text="⚠️ No pude resolver el home remoto para listar. "
                        "Revisá la conexión al cluster.",
                        ok=False,
                    )
                base = f"{home}/{base}"
            raw = base
        try:
            req = ListFilesRequest(path=str(raw))
        except (ValidationError, ValueError):
            return Reply(
                text=f"⚠️ Ruta inválida: {raw!r}. Tiene que ser una ruta "
                "absoluta (empezar con /), sin caracteres especiales ni '..'.",
                ok=False,
            )
        result = ctx.cluster.list_directory(req.path)
        if not result.ok:
            logger.warning("ls remoto falló para %s: %s", req.path, result.message)
            return Reply(text=f"❌ 📂 {result.message}", ok=False)
        return Reply(
            text=f"📂 {req.path}:\n{_truncate_listing(result.message)}",
            monospace=True,
        )

    def _query_history(self, ctx: _Ctx, params: dict) -> Reply:
        try:
            flt = HistoryFilter(
                job_id=params.get("job_id"),
                name_contains=params.get("filtro_busqueda"),
                owner_id=ctx.user_id,  # nunca viene del LLM
            )
        except (ValidationError, ValueError) as exc:
            return Reply(text=f"⚠️ Filtro inválido:\n{exc}", ok=False)
        rows = self._history.search(flt)
        if not rows:
            return Reply(text="📋 Historial:\nNo se encontraron registros.")
        return Reply(text="📋 Historial:\n" + _format_history_table(rows), monospace=True)

    def _modify_structure(self, ctx: _Ctx, params: dict) -> Reply:
        formula = params.get("formula") or params.get("formula_quimica")
        if not formula:
            return Reply(
                text="⚠️ Decime qué estructura querés (fórmula), p. ej.: "
                '"generá un POSCAR de Si diamond 2x2x2".',
                ok=False,
            )
        try:
            kind = (
                StructureKind.MOLECULE
                if str(params.get("tipo_estructura", "")).lower() == "molecule"
                else StructureKind.BULK
            )
            fmt_raw = str(params.get("formato_salida", "vasp")).lower()
            fmt = OutputFormat(fmt_raw) if fmt_raw in OutputFormat._value2member_map_ else OutputFormat.VASP
            sc = params.get("supercelda") or [1, 1, 1]
            req = StructureRequest(
                formula=str(formula),
                kind=kind,
                crystal=params.get("red_cristalina"),
                lattice_a=params.get("parametro_red"),
                supercell=tuple(int(x) for x in sc),
                vacuum=params.get("vacio"),
                output_format=fmt,
                remote_dest_dir=params.get("destino_remoto"),
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return Reply(text=f"⚠️ Parámetros de estructura inválidos:\n{exc}", ok=False)

        try:
            result = self._structures.build(req)
        except Exception as exc:  # StructureBuildError y afines
            return Reply(text=f"⚠️ No pude construir la estructura:\n{exc}", ok=False)

        uploaded_note = ""
        if req.remote_dest_dir:
            remote_path = f"{req.remote_dest_dir.rstrip('/')}/{result.filename}"
            up = ctx.cluster.upload_file(result.local_path, remote_path)
            uploaded_note = (
                f"\n📤 {up.message}" if up.ok else f"\n⚠️ Falló la subida: {up.message}"
            )

        return Reply(text=f"🔬 Estructura generada:\n{result.describe()}{uploaded_note}")

    # ------------------------------------------------------------------
    # Cálculo VASP completo: genera inputs, sube y deja el sbatch listo
    # ------------------------------------------------------------------

    # Orden de búsqueda del pseudopotencial de cada elemento en la
    # biblioteca del cluster (variantes semi-core primero, como recomienda
    # VASP para la mayoría de los metales de transición).
    _POTCAR_VARIANTS = ("_sv", "_pv", "")

    def _prepare_calc(self, ctx: _Ctx, params: dict) -> Reply:
        if self._calc_inputs is None:
            return Reply(
                text="⚠️ La preparación de cálculos VASP no está configurada en este bot."
            )
        formula = params.get("formula") or params.get("formula_quimica")
        if not formula:
            return Reply(
                text="⚠️ Decime qué material querés calcular, p. ej.: "
                '"relajá los parámetros de red del bulk de W".'
            )
        if not self._potcar_dir or not self._potcar_dir.startswith("/"):
            return Reply(
                text="⚠️ Falta configurar la biblioteca de POTCAR del cluster "
                "(BECARIO_POTCAR_DIR, ruta absoluta) para preparar cálculos VASP."
            )

        kind_raw = str(params.get("tipo_calculo") or "").strip().lower()
        lo, hi = params.get("encut_min"), params.get("encut_max")
        if kind_raw in CalcKind._value2member_map_:
            calc_kind = CalcKind(kind_raw)
        elif lo and hi:
            # El LLM a veces extrae el rango pero omite tipo_calculo: un
            # rango de ENCUT solo tiene sentido como barrido.
            calc_kind = CalcKind.ENCUT_SCAN
        else:
            calc_kind = CalcKind.STATIC

        # Barrido de ENCUT explícito (si el usuario dio rango); si no, el
        # modelo de dominio usa su default.
        encut_values = None
        if calc_kind is CalcKind.ENCUT_SCAN and lo and hi:
            step = int(params.get("encut_paso") or 50)
            if step <= 0:
                return Reply(text="⚠️ El paso del barrido de ENCUT debe ser positivo.")
            encut_values = list(range(int(lo), int(hi) + 1, step))

        try:
            sc = params.get("supercelda") or [1, 1, 1]
            kp = params.get("puntos_k")
            req = VaspCalcRequest(
                formula=str(formula),
                crystal=params.get("red_cristalina"),
                lattice_a=params.get("parametro_red"),
                supercell=tuple(int(x) for x in sc),
                calc_kind=calc_kind,
                encut=int(params.get("encut") or 520),
                kpoints=tuple(int(x) for x in kp) if kp else None,
                encut_values=encut_values,
                partition=params.get("particion") or "default",
                nodes=int(params.get("nodos") or 1),
                time_limit=params.get("tiempo_limite") or "01:00:00",
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return Reply(text=f"⚠️ Parámetros de cálculo inválidos:\n{exc}")

        try:
            result = self._calc_inputs.generate(req)
        except Exception as exc:  # StructureBuildError y afines
            return Reply(text=f"⚠️ No pude generar los inputs:\n{exc}")

        # POTCAR: una variante por elemento, en el orden de especies del POSCAR.
        potcar_sources: list[str] = []
        for element in result.elements:
            candidates = [
                f"{self._potcar_dir}/{element}{suffix}/POTCAR"
                for suffix in self._POTCAR_VARIANTS
            ]
            found = next((c for c in candidates if ctx.cluster.file_exists(c)), None)
            if found is None:
                tried = ", ".join(f"{element}{s}" for s in self._POTCAR_VARIANTS)
                return Reply(
                    text=f"⚠️ No encontré POTCAR para {element} en "
                    f"{self._potcar_dir} (busqué {tried})."
                )
            potcar_sources.append(found)

        base = self._remote_base
        if not base.startswith("/"):
            home = ctx.cluster.home_dir()
            if not home:
                return Reply(
                    text="⚠️ No pude resolver el home remoto para ubicar la corrida. "
                    "Revisá la conexión al cluster."
                )
            base = f"{home}/{base}"
        remote_run_dir = f"{base}/{result.run_name}"

        up = ctx.cluster.upload_dir(result.local_dir, remote_run_dir)
        if not up.ok:
            return Reply(text=f"⚠️ Falló la subida de los inputs: {up.message}")

        cat = ctx.cluster.concat_files(potcar_sources, f"{remote_run_dir}/POTCAR")
        if not cat.ok:
            return Reply(text=f"⚠️ No pude armar el POTCAR en el cluster: {cat.message}")

        try:
            slurm_req = SlurmJobRequest(
                job_name=f"{req.formula}_{calc_kind.value}",
                partition=req.partition,
                nodes=req.nodes,
                time_limit=req.time_limit,
                script_path=f"{remote_run_dir}/run_vasp.sh",
            )
        except (ValidationError, ValueError) as exc:
            return Reply(text=f"⚠️ Parámetros inválidos para el envío:\n{exc}")

        fingerprint = _calc_fingerprint(req)
        duplicate_note = self._duplicate_note(ctx.user_id, slurm_req.job_name, fingerprint)

        workflow = "encut_scan" if calc_kind is CalcKind.ENCUT_SCAN else ""
        payload = slurm_req.model_dump()
        payload["workflow"] = workflow
        payload["calc_fingerprint"] = fingerprint
        payload["run_dir"] = remote_run_dir
        action = PendingAction(
            chat_id=ctx.chat_id,
            requester_id=ctx.user_id,
            intent=Intent.SUBMIT_SLURM,
            description=(
                f"🧪 Enviar cálculo VASP ({calc_kind.value}, cuenta "
                f"{ctx.identity.ssh_user}):\n{result.describe()}\n"
                f"📂 {remote_run_dir}\n{slurm_req.describe()}"
            ),
            payload=payload,
            request_intent=Intent.PREPARE_CALC,
            request_params=dict(params),
        )
        plan = PendingPlan(chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=[action])
        token = self._confirmations.put(plan)
        # El aviso de duplicado va PRIMERO: al final de un mensaje largo
        # pasa desapercibido.
        warning_block = f"{duplicate_note}\n\n" if duplicate_note else ""
        return Reply(
            text=(
                f"{warning_block}✅ Inputs generados y subidos al cluster.\n\n"
                f"⚠️ ¿Confirmás el envío?\n\n{action.description}"
            ),
            needs_confirmation=True,
            confirmation_token=token,
            allow_modify=True,
        )

    def _duplicate_note(self, owner_id: int, job_name: str, fingerprint: str) -> str:
        """Aviso si esto (o algo muy similar) ya se corrió: mismo job_name =
        mismo material y tipo de cálculo; misma huella = pedido idéntico."""
        if self._calc_runs is None:
            return ""
        rows = self._calc_runs.find_by_name(owner_id, job_name)
        if not rows:
            return ""
        exact = next((r for r in rows if r.get("fingerprint") == fingerprint), None)
        if exact is not None:
            header = (
                f"🔁 OJO: ESTO YA SE CORRIÓ, con exactamente las mismas "
                f"condiciones, el {exact.get('fecha', '?')} "
                f"(job {exact.get('job_id', '?')}):\n📂 {exact.get('run_dir', '?')}"
            )
        else:
            prev = rows[0]
            header = (
                f"🔁 OJO: ya corriste algo muy similar (mismo material y "
                f"tipo de cálculo) el {prev.get('fecha', '?')} "
                f"(job {prev.get('job_id', '?')}):\n📂 {prev.get('run_dir', '?')}"
            )
        return header + (
            "\n¿Lo corro de todas formas? Confirmá para correrlo igual, o "
            "tocá ✏️ Modificar para usar este plan de base y cambiar algo."
        )

    # ------------------------------------------------------------------
    # Resultados de corridas previas (parámetros de red, energía)
    # ------------------------------------------------------------------
    def _query_results(self, ctx: _Ctx, params: dict) -> Reply:
        if self._calc_runs is None:
            return Reply(
                text="⚠️ La consulta de resultados no está configurada en este bot.",
                ok=False,
            )

        formula = params.get("formula") or params.get("formula_quimica")
        prefix = f"{str(formula).strip()}_" if formula else ""
        rows = self._calc_runs.find_recent(ctx.user_id, prefix)
        if not rows:
            de = f" de {formula}" if formula else ""
            return Reply(
                text=f"📭 No encontré corridas tuyas{de} registradas. "
                "Pedime el cálculo y lo corremos.",
                ok=False,
            )
        # Los parámetros de red relajados salen de una relajación; si no
        # hay, se usa la corrida más reciente que haya (celda de entrada).
        row = next(
            (r for r in rows if CalcKind.RELAX.value in str(r.get("job_name", ""))),
            rows[0],
        )
        run_dir = str(row.get("run_dir", "")).rstrip("/")
        if not run_dir:
            return Reply(text="⚠️ La corrida registrada no tiene directorio asociado.", ok=False)

        cell_text = ctx.cluster.read_file(f"{run_dir}/CONTCAR")
        source = "CONTCAR (celda relajada)"
        if not (cell_text and cell_text.strip()):
            cell_text = ctx.cluster.read_file(f"{run_dir}/POSCAR")
            source = "POSCAR (celda de entrada, sin relajar)"
        if not (cell_text and cell_text.strip()):
            # Barrido: los inputs viven en los subdirectorios encut_*.
            for name in sorted(ctx.cluster.list_dir(run_dir) or []):
                if name.startswith("encut_"):
                    cell_text = ctx.cluster.read_file(f"{run_dir}/{name}/POSCAR")
                    if cell_text:
                        source = f"{name}/POSCAR (celda de entrada, sin relajar)"
                        break
        cell = _parse_poscar_cell(cell_text)
        if cell is None:
            return Reply(
                text=f"⚠️ No pude leer la celda de la corrida "
                f"{row.get('job_name', '?')} (job {row.get('job_id', '?')}) en:\n"
                f"📂 {run_dir}\n¿Sigue existiendo en el cluster?",
                ok=False,
            )

        a, b, c, alpha, beta, gamma = cell
        lines = [
            f"📐 Parámetros de red — {row.get('job_name', '?')} "
            f"(job {row.get('job_id', '?')}, {row.get('fecha', '?')}):",
            f"a = {a:.4f} Å,  b = {b:.4f} Å,  c = {c:.4f} Å",
            f"α = {alpha:.2f}°,  β = {beta:.2f}°,  γ = {gamma:.2f}°",
            f"(fuente: {source})",
        ]
        energy = _parse_last_e0(ctx.cluster.read_file(f"{run_dir}/OSZICAR"))
        if energy is not None:
            lines.append(f"E0 = {energy:.6f} eV")
        lines.append(f"📂 {run_dir}")
        return Reply(text="\n".join(lines))
