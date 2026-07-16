"""Handlers de trabajos Slurm: estado, envío y cancelación.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
Los `prepare_*` arman la confirmación; los `execute_*` corren recién al
confirmar (mirror que en la fachada exponen `_intent_handlers()` y
`_step_executors()`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import ValidationError

from ...domain.models import (
    Intent,
    JobId,
    PendingAction,
    PendingPlan,
    SlurmJobRequest,
    TrackedJob,
)
from ..context import Reply, _Ctx

if TYPE_CHECKING:
    from ..services import BecarioService


def check_status(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
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


def build_submit_action(svc: "BecarioService", ctx: _Ctx, params: dict) -> PendingAction | Reply:
    """Arma el `PendingAction` de un envío Slurm, o el `Reply` de error
    si los parámetros no validan. Separado de `prepare_submit` para
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


def prepare_submit(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    built = build_submit_action(svc, ctx, params)
    if isinstance(built, Reply):
        return built
    plan = PendingPlan(chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=[built])
    token = svc._confirmations.put(plan)
    return Reply(
        text=f"⚠️ ¿Confirmás esta acción?\n\n{built.description}",
        needs_confirmation=True,
        confirmation_token=token,
        allow_modify=True,
    )


def build_cancel_action(svc: "BecarioService", ctx: _Ctx, params: dict) -> PendingAction | Reply:
    """Ídem `build_submit_action`, para cancelar_calculo."""
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


def prepare_cancel(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    built = build_cancel_action(svc, ctx, params)
    if isinstance(built, Reply):
        return built
    plan = PendingPlan(chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=[built])
    token = svc._confirmations.put(plan)
    return Reply(
        text=f"⚠️ ¿Confirmás esta acción?\n\n{built.description}",
        needs_confirmation=True,
        confirmation_token=token,
    )


def execute_submit(svc: "BecarioService", ctx: _Ctx, action: PendingAction) -> tuple[bool, str]:
    # `workflow` viaja junto a los campos del job pero no es parte de
    # SlurmJobRequest (Pydantic ignora las claves extra del payload).
    req = SlurmJobRequest(**action.payload)
    result = ctx.cluster.submit_job(req)
    if result.ok and result.job_id:
        svc._job_tracker.track(TrackedJob(
            job_id=result.job_id,
            owner_id=ctx.user_id,
            chat_id=action.chat_id,
            ssh_user=ctx.identity.ssh_user,
            job_name=req.job_name,
            script_path=req.script_path,
            workflow=action.payload.get("workflow", ""),
        ))
        svc._history.add(
            owner_id=ctx.user_id, job_id=result.job_id,
            nombre_trabajo=req.job_name, estado="enviado",
        )
        # Cálculos VASP: registrar la huella para poder avisar si
        # se vuelve a pedir lo mismo (o algo muy similar).
        if svc._calc_runs is not None and action.payload.get("calc_fingerprint"):
            svc._calc_runs.add(
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


def execute_cancel(svc: "BecarioService", ctx: _Ctx, action: PendingAction) -> tuple[bool, str]:
    jid = JobId(value=action.payload["job_id"])
    result = ctx.cluster.cancel_job(jid)
    if result.ok:
        # Ya lo sabe (lo canceló él mismo): evita el aviso duplicado
        # del monitor cuando vea el estado CANCELLED más tarde. Como el
        # monitor tampoco lo va a registrar, queda asentado acá.
        svc._job_tracker.mark_notified(jid.value, ctx.user_id)
        svc._history.add(
            owner_id=ctx.user_id, job_id=jid.value,
            nombre_trabajo="", estado="cancelado",
        )
    status = "✅" if result.ok else "❌"
    return result.ok, f"{status} 🛑 {result.message}"
