"""Coordinación de planes: plan compuesto (ADR-0006), batch (ADR-0007) y
ediciones pendientes.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
Todo lo que antes era `self._x` sigue siendo `svc._x` (nunca la función de
este módulo directa): los tests leen y reasignan atributos de la instancia,
y el ciclo edición → `_dispatch_plan` tiene que seguir pasando por ella.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from ..domain.models import (
    Intent,
    PendingAction,
    PendingPlan,
    Plan,
    PlanStep,
)
from .context import Reply, _Ctx
from .handlers import jobs
from .plan_executor import PlanExecutor
from .textos import HELP_TEXT

if TYPE_CHECKING:
    from .services import BecarioService

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Composición multi-paso (planes de más de un paso, ver ADR-0006)
# ----------------------------------------------------------------------
def materialize_step_fn(
    svc: "BecarioService",
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
    handler = svc._intent_handlers().get(step.action)

    def _run() -> tuple[bool, str]:
        if handler is None or step.action not in svc._MATERIALIZABLE_STEP_INTENTS:
            return False, f"el paso '{step.action.value}' no se puede combinar en un plan"
        reply = handler(ctx, step.parametros)
        if reply.awaiting_params and awaiting_out is not None:
            awaiting_out.append(index)
        return reply.ok, reply.text

    return _run


def run_composite_plan(svc: "BecarioService", ctx: _Ctx, plan: Plan) -> Reply:
    """Un plan de más de un paso: los pasos no destructivos materializan
    EN ORDEN al construir el plan (stop-on-failure); si el plan termina
    en un paso destructivo (`enviar_slurm`/`cancelar_calculo`, único
    posible por `Plan._v_destructive_last`), la confirmación se pide
    SOLO para esa cola — el resto ya corrió y se muestra ejecutado."""
    # Acá NO llega ningún plan con `preparar_calculo`: todo plan
    # multi-paso que contenga uno se confirma entero como batch
    # (`_is_batch_plan`). La única forma de que un cálculo llegue hasta
    # este método sería junto a un paso destructivo, y `_v_destructive_last`
    # lo obliga a ir último — o sea, el cálculo quedaría en el prefijo y
    # cae en el fail-closed de abajo.
    #
    # Antes vivía acá una "cola de cálculos" que materializaba el prefijo
    # y emitía un followup con confirmación individual por cálculo. Quedó
    # inalcanzable al enrutar estos planes al batch, y se sacó en vez de
    # dejarla: código muerto que describe un comportamiento que ya no
    # existe es peor que no tener nada, porque se lee como si rigiera.
    tail = plan.steps[-1]
    has_destructive_tail = tail.action in Intent.destructive()
    prefix = plan.steps[:-1] if has_destructive_tail else plan.steps

    # `preparar_calculo` nunca es materializable (deja su propio pedido
    # pendiente) — si aparece en `prefix` esto también lo atrapa,
    # incluso sin cola destructiva (ej. `[preparar_calculo, crear_dir]`,
    # donde `tail` termina siendo el propio `preparar_calculo`).
    unsupported = [s for s in prefix if s.action not in svc._MATERIALIZABLE_STEP_INTENTS]
    if unsupported:
        # v1 no combina preparar_calculo (ni ningún otro paso "de
        # preparación") dentro de un plan multi-paso — fail-closed,
        # sin ejecutar nada (ver deviations, tarea 4.3).
        return Reply(text=HELP_TEXT)

    awaiting: list[int] = []
    step_fns = [
        svc._materialize_step_fn(ctx, s, i, awaiting)
        for i, s in enumerate(prefix, start=1)
    ]
    result = PlanExecutor().run(step_fns)
    if awaiting:
        svc._arm_pending_edit(
            ctx.user_id, ctx.chat_id, svc._plan_steps_as_requests(plan),
            awaiting_index=awaiting[0],
        )

    if not has_destructive_tail:
        return Reply(text=result.report, awaiting_params=bool(awaiting))

    if not result.ok:
        lines = result.report_lines + [f"{len(plan.steps)}. ⏸ omitido"]
        return Reply(text="\n".join(lines), awaiting_params=bool(awaiting))

    return svc._prepare_destructive_tail(ctx, tail, prefix, result.report_lines)


def prepare_destructive_tail(
    svc: "BecarioService",
    ctx: _Ctx,
    step: PlanStep,
    prefix_steps: list[PlanStep],
    prefix_report: list[str],
) -> Reply:
    """Arma la confirmación de la cola destructiva final, después de
    que el resto del plan ya materializó sin errores."""
    builder = {
        Intent.SUBMIT_SLURM: jobs.build_submit_action,
        Intent.CANCEL_JOB: jobs.build_cancel_action,
    }[step.action]
    built = builder(svc, ctx, step.parametros)
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
    token = svc._confirmations.put(plan)
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
