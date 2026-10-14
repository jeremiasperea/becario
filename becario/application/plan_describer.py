"""Funciones puras sobre planes: describirlos, clasificarlos y pasarlos a
pedidos.

Salieron de `BecarioService` porque no tocan estado de la instancia.
La clase las conserva como alias (`BecarioService._describe_batch_step`,
etc.): tests y llamadores internos siguen pasando por el atributo de la
clase, así que parchearlo ahí sigue teniendo efecto. Lo que antes se leía
de `cls` (marcadores, intents materializables) llega acá por parámetro.
"""
from __future__ import annotations

from typing import Callable, Iterable

from ..domain.models import (
    _MATERIAL_INTENTS,
    _MAX_AUTOMATERIALIZE_STEPS,
    Intent,
    Plan,
    PlanStep,
    StructureKind,
)


def looks_compound(text: str, markers: Iterable[str]) -> bool:
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
    hits = sum(lowered.count(m) for m in markers)
    return hits >= 1 and len(words) >= 10


def needs_decomposition(plan: Plan) -> bool:
    """Firma exacta del fallo medido: plan multi-paso con algún
    cálculo sin material."""
    if len(plan.steps) < 2:
        return False
    return any(
        s.action is Intent.PREPARE_CALC and not s.parametros.get("formula")
        for s in plan.steps
    )


def is_batch_plan(plan: Plan, materializable: Iterable[Intent]) -> bool:
    """Un plan es BATCH (ADR-0007) si excede la composición chica que se
    auto-materializa —— más de `_MAX_AUTOMATERIALIZE_STEPS` pasos O varios
    cálculos —— y está formado SOLO por pasos materializables y
    `preparar_calculo` (sin cola realmente destructiva). Estos planes
    salen del descompositor; se confirman enteros antes de tocar el
    cluster, así que nada se auto-ejecuta y el cap deja de gatear el
    blast-radius (lo hace la confirmación)."""
    n_calc = sum(1 for s in plan.steps if s.action is Intent.PREPARE_CALC)
    if not n_calc and len(plan.steps) <= _MAX_AUTOMATERIALIZE_STEPS:
        return False
    return all(
        s.action in materializable or s.action is Intent.PREPARE_CALC
        for s in plan.steps
    )


def describe_structure_step(p: dict) -> str:
    """Una línea para un paso de estructura: material primero, después
    lo que lo distingue de otra estructura del mismo material."""
    formula = p.get("formula") or p.get("formula_quimica") or "?"
    detalles = []
    if (kind := p.get("tipo_estructura")) and kind != StructureKind.BULK.value:
        detalles.append(str(kind))
    if miller := p.get("miller"):
        detalles.append("(" + "".join(str(i) for i in miller) + ")")
    if capas := p.get("capas"):
        detalles.append(f"{capas} capas")
    if red := p.get("red_cristalina"):
        detalles.append(str(red))
    if celda := p.get("supercelda"):
        detalles.append("×".join(str(n) for n in celda))
    if fuente := p.get("fuente_estructura"):
        detalles.append(f"desde {fuente}")
    cola = f" ({', '.join(detalles)})" if detalles else ""
    return f"🧱 generar estructura de {formula}{cola}"


def describe_batch_step(
    step: PlanStep,
    describe_structure: Callable[[dict], str] = describe_structure_step,
) -> str:
    """Qué va a hacer este paso, en una línea, para el preview del batch.

    No es cosmética: en un batch NADA se ejecuta hasta que el usuario
    confirma, así que esta línea es lo ÚNICO sobre lo que decide. Un
    `• listar_archivos` pelado —el nombre crudo del enum— le pide que
    apruebe a ciegas.

    Cada rama nombra los parámetros que el handler correspondiente
    realmente lee; cuando un parámetro falta y el sistema resuelve un
    default, la línea dice cuál es ese default en vez de callarlo.
    """
    p = step.parametros
    if step.action is Intent.CREATE_DIR:
        return f"📁 crear carpeta {p.get('destino_remoto', '?')}"
    if step.action is Intent.LIST_FILES:
        # Sin ruta, `list_files` lista la base de corridas.
        return f"📂 listar {p.get('destino_remoto') or 'tus corridas'}"
    if step.action is Intent.VIEW_FILE:
        # `view_file` acepta nombre suelto (resuelto contra la última
        # corrida) o ruta absoluta, nunca los dos.
        if destino := p.get("destino_remoto"):
            return f"👁 ver {destino}"
        if nombre := p.get("nombre_archivo"):
            return f"👁 ver {nombre} (de tu última corrida)"
        return "👁 ver un archivo (falta cuál)"
    if step.action is Intent.MODIFY_STRUCTURE:
        return describe_structure(p)
    if step.action is Intent.QUERY_RESULTS:
        formula = p.get("formula") or p.get("formula_quimica")
        return f"📈 resultados de {formula}" if formula else "📈 resultados del último cálculo"
    if step.action is Intent.QUERY_DB:
        if filtro := p.get("filtro_busqueda"):
            return f"📊 historial que contenga {filtro!r}"
        return "📊 historial de cálculos"
    if step.action is Intent.CHECK_STATUS:
        if job := p.get("job_id"):
            return f"🔎 estado del trabajo {job}"
        return "🔎 estado de tus trabajos en cola"
    return f"• {step.action.value}"


def plan_steps_as_requests(plan: Plan) -> list[tuple[Intent, dict]]:
    """El plan como lista de pedidos originales, que es lo que guarda
    `_PendingEdit`: re-armar SIEMPRE parte de acá, nunca de un payload
    ya materializado."""
    return [(s.action, dict(s.parametros)) for s in plan.steps]


def propagate_answer(
    steps: list[tuple[Intent, dict]],
    answered_index: int,
    base_params: dict,
    new_params: dict,
) -> None:
    """Lleva el dato recién contestado a los pasos posteriores que hablan
    del MISMO material y tienen el mismo hueco. Muta `steps` in place.

    "Armá el slab de ZrO2 y relajalo" son dos pasos sobre una sola
    superficie: si al usuario le faltó la cara, le falta a los dos. Sin
    esto, contesta «(001)», el paso 1 se completa y el paso 2 le vuelve
    a preguntar lo mismo con las mismas palabras — se ve como si el bot
    lo hubiera ignorado.

    Tres condiciones para no pisar un pedido legítimo:

    - Solo pasos POSTERIORES al contestado (los previos ya se
      resolvieron con lo que el usuario había dicho entonces).
    - Solo pasos sobre el mismo material: `formula` igual y ambos de
      `_MATERIAL_INTENTS`. Dos materiales distintos son dos pedidos.
    - Solo claves que el paso NO define ya. "Generá el POSCAR de Si
      (001) y relajá Si (110)" son dos caras dichas a propósito.
    """
    material = base_params.get("formula") or base_params.get("formula_quimica")
    if not material:
        return
    for i in range(answered_index, len(steps)):
        intent, params = steps[i]
        if intent not in _MATERIAL_INTENTS:
            continue
        if (params.get("formula") or params.get("formula_quimica")) != material:
            continue
        faltantes = {k: v for k, v in new_params.items() if k not in params}
        if faltantes:
            steps[i] = (intent, {**params, **faltantes})


def render_plan_context(steps: list[tuple[Intent, dict]]) -> str:
    """Enumeración legible del plan pendiente: se la mandamos al LLM
    de `extract_edit` como contexto para el targeting semántico, y
    se la mostramos al usuario cuando el targeting sale ambiguo."""
    lines = []
    for i, (intent, params) in enumerate(steps, start=1):
        resumen = ", ".join(f"{k}={v}" for k, v in params.items()) or "sin parámetros"
        lines.append(f"{i}. {intent.value} ({resumen})")
    return "\n".join(lines)
