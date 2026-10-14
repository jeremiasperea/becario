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
from .handlers import calc, jobs
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


# ----------------------------------------------------------------------
# Batch (ADR-0007): se stagea entero y se ejecuta todo al confirmar
# ----------------------------------------------------------------------
def prepare_batch(svc: "BecarioService", ctx: _Ctx, plan: Plan) -> Reply:
    """Batch (ADR-0007): valida y describe cada paso SIN tocar el cluster,
    stagea el plan entero con una sola confirmación. Recién al confirmar
    se ejecuta todo. Fail-closed: un cálculo inválido aborta el batch sin
    efectos."""
    actions: list[PendingAction] = []
    preview: list[str] = []
    n_calc = 0
    for i, step in enumerate(plan.steps, start=1):
        if step.action is Intent.PREPARE_CALC:
            req = calc._build_calc_request(svc, step.parametros)
            if isinstance(req, Reply):
                # Cálculo inválido: nada se ejecuta. Si lo que falta es un
                # dato pedible, el batch entero queda esperando en vez de
                # obligar a rearmar los N pasos.
                svc._arm_if_awaiting(ctx, plan, i, req)
                return req
            # La ambigüedad de fase de Materials Project (¿tetragonal o
            # monoclínica?) es el mismo caso que la cara de la losa: si
            # se descubre recién al EJECUTAR el batch ya confirmado, la
            # repregunta pierde el `awaiting_params` en el camino
            # (`execute_calc`) y no queda nada esperando la respuesta.
            # Se adelanta acá, antes de stagear — la consulta a MP es
            # de solo lectura, no toca el cluster.
            pregunta_fase = calc.phase_question(svc, req)
            if pregunta_fase is not None:
                svc._arm_if_awaiting(ctx, plan, i, pregunta_fase)
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
                    svc._arm_if_awaiting(ctx, plan, i, problema)
                    return problema
            line = svc._describe_batch_step(step)
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
    token = svc._confirmations.put(pending)
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


def execute_batch(
    svc: "BecarioService", plan: PendingPlan, requester_id: int, identity, cluster,
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
            reply = svc._execute_batch_step(ctx, action)
        except Exception:
            # Un paso que revienta corta el batch igual que uno que
            # falla (misma semántica sin-rollback de ADR-0006): lo ya
            # ejecutado queda hecho y los que siguen se omiten. La
            # diferencia es que de ESTE paso no sabemos el desenlace.
            reply = svc._fallo_tras_confirmar(action)
        # El texto del handler ya trae su propio ✅/🚀/⚠️; solo lo
        # numeramos (evita duplicar el mark).
        lines.append(f"{i}. {reply.text}")
        ok_all = ok_all and reply.ok
    return Reply(text="\n".join(lines), ok=ok_all)

def execute_batch_step(svc: "BecarioService", ctx: _Ctx, action: PendingAction) -> Reply:
    """Ejecuta un paso de batch al confirmar: `preparar_calculo` genera,
    sube y ENVÍA de una (la confirmación del batch ya lo avaló); el resto
    usa su handler normal, que materializa en el acto."""
    if action.intent is Intent.PREPARE_CALC:
        return calc.execute_calc(svc, ctx, action.payload)
    handler = svc._intent_handlers().get(action.intent)
    if handler is None:  # pragma: no cover - defensivo
        return Reply(text=f"paso '{action.intent.value}' no ejecutable", ok=False)
    return handler(ctx, action.payload)
