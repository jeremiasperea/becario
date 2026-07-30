"""Handlers de estructuras y cálculos VASP: genera inputs, sube y prepara.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ...domain.models import (
    Axis,
    CalcKind,
    Intent,
    OutputFormat,
    PendingAction,
    PendingPlan,
    SlurmJobRequest,
    StructureKind,
    StructureQuery,
    StructureRequest,
    StructureResolutionError,
    StructureResolutionReason,
    StructureSource,
    VaspCalcRequest,
    elements_of,
)
from ..context import Reply, _Ctx

if TYPE_CHECKING:
    from ..services import BecarioService


def _ask_for_miller(formula: str) -> Reply:
    """Falta la cara de la losa: se pregunta, no se adivina.

    Elegir la orientación por el usuario no es un default cómodo: cambia la
    terminación, la polaridad y los estados de superficie. Una (001) y una
    (111) del mismo material son experimentos distintos, así que un default
    silencioso devolvería un resultado creíble y equivocado.
    """
    return Reply(
        text=(
            f"🔭 Para armar la superficie de {formula} necesito la cara: "
            "¿(001), (110), (111)…?\n"
            "Es lo único que me falta — el resto del pedido lo tengo."
        ),
        ok=False,
        awaiting_params=True,
    )


def _slab_params(params: dict, kind: StructureKind) -> dict:
    """Parte de losa de los parámetros crudos del router.

    Devuelve `{}` para lo que no es una losa: los modelos RECHAZAN miller y
    capas en un bulk, así que un `tipo_estructura` mal extraído no arrastra
    campos que no corresponden. Solo se traduce lo que el mensaje trajo — el
    índice de Miller ausente lo resuelve el dominio, no un default acá.
    """
    if kind is not StructureKind.SLAB:
        return {}
    out: dict = {}
    miller = params.get("miller")
    if miller:
        out["miller"] = tuple(int(x) for x in miller)
    if params.get("capas"):
        out["layers"] = int(params["capas"])
    axis_raw = str(params.get("eje_vacio") or "").lower()
    if axis_raw in Axis._value2member_map_:
        out["vacuum_axis"] = Axis(axis_raw)
    return out


def _calc_fingerprint(req: VaspCalcRequest) -> str:
    """JSON canónico de un pedido de cálculo: dos pedidos con la misma
    huella son EL MISMO cálculo (el rango default del barrido se resuelve
    para que 'sin rango' y 'con el rango default' den la misma huella)."""
    data = req.model_dump()
    if req.calc_kind is CalcKind.ENCUT_SCAN:
        data["encut_values"] = req.scan_values()
    return json.dumps(data, sort_keys=True, default=str)


def modify_structure(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    formula = params.get("formula") or params.get("formula_quimica")
    if not formula:
        return Reply(
            text="⚠️ Decime qué estructura querés (fórmula), p. ej.: "
            '"generá un POSCAR de Si diamond 2x2x2".',
            ok=False,
        )
    try:
        kind_raw = str(params.get("tipo_estructura", "")).lower()
        kind = (
            StructureKind(kind_raw)
            if kind_raw in StructureKind._value2member_map_
            else StructureKind.BULK
        )
        if kind is StructureKind.SLAB and not params.get("miller"):
            return _ask_for_miller(str(formula))
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
            **_slab_params(params, kind),
        )
    except (ValidationError, ValueError, TypeError) as exc:
        return Reply(text=f"⚠️ Parámetros de estructura inválidos:\n{exc}", ok=False)

    try:
        result = svc._structures.build(req)
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


@dataclass
class _PreparedCalc:
    """Resultado de preparar un cálculo (generar + subir + armar el envío),
    listo para stagear una confirmación (camino simple) o para enviar de una
    (paso de un batch confirmado)."""

    payload: dict          # SlurmJobRequest.model_dump() + workflow/fingerprint/run_dir
    description: str        # detalle humano del envío
    duplicate: str         # aviso de duplicado (vacío si no hay)


def _build_calc_request(svc: "BecarioService", params: dict) -> "VaspCalcRequest | Reply":
    """Valida y arma el `VaspCalcRequest` desde los parámetros crudos. SIN
    I/O al cluster: sirve tanto para el preview del batch (describir sin
    tocar nada) como para el camino que después genera y sube."""
    if svc._calc_inputs is None:
        return Reply(
            text="⚠️ La preparación de cálculos VASP no está configurada en este bot."
        )
    formula = params.get("formula") or params.get("formula_quimica")
    if not formula:
        return Reply(
            text="⚠️ Decime qué material querés calcular, p. ej.: "
            '"relajá los parámetros de red del bulk de W".'
        )
    if not svc._potcar_dir or not svc._potcar_dir.startswith("/"):
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

    # Fuente de estructura (Materials Project): el router puede extraer un
    # `mp_id` explícito y/o forzar el origen vía `fuente_estructura`. Un
    # valor de fuente fuera del enum se ignora (queda AUTO) para no romper
    # el cálculo por un typo del LLM; `mp_id` inválido sí lo caza el modelo.
    src_raw = str(params.get("fuente_estructura") or "").strip().lower()
    source = (
        StructureSource(src_raw)
        if src_raw in StructureSource._value2member_map_
        else StructureSource.AUTO
    )

    kind_raw = str(params.get("tipo_estructura", "")).lower()
    struct_kind = (
        StructureKind.SLAB if kind_raw == StructureKind.SLAB.value else StructureKind.BULK
    )
    if struct_kind is StructureKind.SLAB and not params.get("miller"):
        return _ask_for_miller(str(formula))

    try:
        sc = params.get("supercelda") or [1, 1, 1]
        kp = params.get("puntos_k")
        return VaspCalcRequest(
            formula=str(formula),
            crystal=params.get("red_cristalina"),
            lattice_a=params.get("parametro_red"),
            supercell=tuple(int(x) for x in sc),
            kind=struct_kind,
            vacuum=params.get("vacio"),
            **_slab_params(params, struct_kind),
            calc_kind=calc_kind,
            encut=int(params.get("encut") or 520),
            kpoints=tuple(int(x) for x in kp) if kp else None,
            encut_values=encut_values,
            mp_id=params.get("mp_id"),
            source=source,
            partition=params.get("particion") or "default",
            nodes=int(params.get("nodos") or 1),
            time_limit=params.get("tiempo_limite") or "01:00:00",
        )
    except (ValidationError, ValueError, TypeError) as exc:
        return Reply(text=f"⚠️ Parámetros de cálculo inválidos:\n{exc}")


def describe_calc_request(req: "VaspCalcRequest") -> str:
    """Línea de preview de un cálculo SIN tocar el cluster (para el batch)."""
    crystal = f" {req.crystal}" if req.crystal else ""
    return (
        f"🧪 {req.formula}{crystal} — {req.calc_kind.value} "
        f"(partición {req.partition}, {req.nodes} nodo(s), {req.time_limit})"
    )


def _resolve_structure(
    svc: "BecarioService", req: "VaspCalcRequest",
) -> "tuple[object, str] | Reply":
    """Decide la FUENTE de la estructura y la resuelve, SIN escribir nada.

    - Elemento simple o `source=ase` -> ASE: devuelve `(None, "")` y el
      generador arma la estructura desde la fórmula (comportamiento previo, R1).
    - Compuesto, `source=mp` o `mp_id` -> Materials Project (R2/R3/R4),
      fail-closed si falta la key (R5) o si MP falla (R6/R7).

    Devuelve `(atoms|None, nota)` en éxito, o un `Reply` de error. La
    resolución ocurre ANTES de crear el directorio de la corrida, así un
    fallo de MP nunca deja una corrida a medias."""
    elements = elements_of(req.formula)
    wants_mp = (
        req.mp_id is not None
        or req.source is StructureSource.MP
        or (req.source is StructureSource.AUTO and len(elements) > 1)
    )
    if not wants_mp:
        return None, ""  # camino ASE: el generador la arma (sin key)

    if svc._structure_provider is None or not svc._mp_api_key:
        return Reply(
            text="⚠️ Para calcular compuestos uso Materials Project, pero falta "
            "configurar la API key (BECARIO_MP_API_KEY)."
        )

    if req.mp_id:
        query = StructureQuery(mp_id=req.mp_id)
    elif len(elements) > 1:
        query = StructureQuery(formula=req.formula, elements=tuple(elements))
    else:
        query = StructureQuery(formula=req.formula)

    try:
        resolution = svc._structure_provider.resolve(query)
    except StructureResolutionError as exc:
        return _mp_error_reply(exc)
    return resolution.atoms, _mp_note(resolution)


def _mp_error_reply(exc: StructureResolutionError) -> Reply:
    """Mapea la causa clasificada a un ⚠️ claro (R6/R7)."""
    if exc.reason is StructureResolutionReason.NO_MATCH:
        text = "⚠️ No encontré una estructura en Materials Project para ese material."
    elif exc.reason is StructureResolutionReason.NETWORK:
        text = "⚠️ No pude conectarme a Materials Project. Probá de nuevo en un rato."
    else:
        text = "⚠️ Materials Project devolvió un error al buscar la estructura."
    return Reply(text=text)


def _mp_note(resolution) -> str:
    """Nota humana con el material elegido (el más estable). Las alternativas
    se listan con su mp-id: el router ya extrae `mp_id`/`fuente_estructura`
    y `_build_calc_request` los reenvía, así que pedir otro polimorfo por su
    id es una interacción real — la nota puede ofrecerla honestamente."""
    head = (
        f"🧬 Estructura de Materials Project: {resolution.formula} "
        f"({resolution.mp_id}, {resolution.spacegroup})"
    )
    if resolution.energy_above_hull is None:
        # Pedida por su mp-id: no conocemos el hull, así que no afirmamos
        # estabilidad — el usuario eligió este polimorfo a propósito.
        lines = [f"{head} — elegida por su mp-id."]
    else:
        lines = [
            f"{head} — la más estable "
            f"(E_hull={resolution.energy_above_hull:.3f} eV/át.)."
        ]
    if resolution.alternatives:
        alts = "; ".join(
            f"{a.formula} {a.mp_id} (E_hull={a.energy_above_hull:.3f} eV/át.)"
            for a in resolution.alternatives
        )
        lines.append(
            f"Otras opciones que devolvió MP: {alts}. Se usa automáticamente "
            "la más estable; si querés otra, pedímela por su mp-id "
            "(p. ej. «usá mp-19770»)."
        )
    return "\n".join(lines)


def _generate_and_upload(
    svc: "BecarioService", ctx: _Ctx, req: "VaspCalcRequest",
) -> "_PreparedCalc | Reply":
    """Genera los inputs, resuelve POTCAR, sube la corrida al cluster y arma
    el envío. TODO el I/O al cluster vive acá. Devuelve un `_PreparedCalc` o
    un `Reply` de error (fail-closed)."""
    # Resolver la estructura (ASE local vs Materials Project) ANTES de generar:
    # un fallo de MP corta acá, sin dejar directorio de corrida a medias.
    resolved = _resolve_structure(svc, req)
    if isinstance(resolved, Reply):
        return resolved
    atoms, mp_note = resolved

    try:
        result = svc._calc_inputs.generate(req, atoms)
    except Exception as exc:  # StructureBuildError y afines
        return Reply(text=f"⚠️ No pude generar los inputs:\n{exc}")

    # POTCAR: una variante por elemento, en el orden de especies del POSCAR.
    potcar_sources: list[str] = []
    for element in result.elements:
        candidates = [
            f"{svc._potcar_dir}/{element}{suffix}/POTCAR"
            for suffix in svc._POTCAR_VARIANTS
        ]
        found = next((c for c in candidates if ctx.cluster.file_exists(c)), None)
        if found is None:
            tried = ", ".join(f"{element}{s}" for s in svc._POTCAR_VARIANTS)
            return Reply(
                text=f"⚠️ No encontré POTCAR para {element} en "
                f"{svc._potcar_dir} (busqué {tried})."
            )
        potcar_sources.append(found)

    base = svc._remote_base
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
            job_name=f"{req.formula}_{req.calc_kind.value}",
            partition=req.partition,
            nodes=req.nodes,
            time_limit=req.time_limit,
            script_path=f"{remote_run_dir}/run_vasp.sh",
        )
    except (ValidationError, ValueError) as exc:
        return Reply(text=f"⚠️ Parámetros inválidos para el envío:\n{exc}")

    fingerprint = _calc_fingerprint(req)
    duplicate = duplicate_note(svc, ctx.user_id, slurm_req.job_name, fingerprint)

    payload = slurm_req.model_dump()
    payload["workflow"] = "encut_scan" if req.calc_kind is CalcKind.ENCUT_SCAN else ""
    payload["calc_fingerprint"] = fingerprint
    payload["run_dir"] = remote_run_dir
    mp_block = f"{mp_note}\n" if mp_note else ""
    description = (
        f"🧪 Enviar cálculo VASP ({req.calc_kind.value}, cuenta "
        f"{ctx.identity.ssh_user}):\n{mp_block}{result.describe()}\n"
        f"📂 {remote_run_dir}\n{slurm_req.describe()}"
    )
    return _PreparedCalc(payload=payload, description=description, duplicate=duplicate)


def prepare_calc(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    """Camino de un solo cálculo (ADR-0006): genera y sube los inputs, y deja
    el ENVÍO pendiente de confirmación individual — comportamiento sin cambios."""
    req = _build_calc_request(svc, params)
    if isinstance(req, Reply):
        return req
    prepared = _generate_and_upload(svc, ctx, req)
    if isinstance(prepared, Reply):
        return prepared

    action = PendingAction(
        chat_id=ctx.chat_id,
        requester_id=ctx.user_id,
        intent=Intent.SUBMIT_SLURM,
        description=prepared.description,
        payload=prepared.payload,
        request_intent=Intent.PREPARE_CALC,
        request_params=dict(params),
    )
    plan = PendingPlan(
        chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=[action],
        decision_id=ctx.decision_id,
    )
    token = svc._confirmations.put(plan)
    # El aviso de duplicado va PRIMERO: al final de un mensaje largo
    # pasa desapercibido.
    warning_block = f"{prepared.duplicate}\n\n" if prepared.duplicate else ""
    return Reply(
        text=(
            f"{warning_block}✅ Inputs generados y subidos al cluster.\n\n"
            f"⚠️ ¿Confirmás el envío?\n\n{prepared.description}"
        ),
        needs_confirmation=True,
        confirmation_token=token,
        allow_modify=True,
    )


def execute_calc(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    """Paso de cálculo de un batch YA confirmado (ADR-0007): genera, sube y
    ENVÍA de una, sin confirmación intermedia (la del batch ya la cubrió).
    Fail-closed: cualquier error corta el batch (`ok=False`)."""
    from . import jobs  # import local: evita ciclo handlers <-> handlers

    req = _build_calc_request(svc, params)
    if isinstance(req, Reply):
        return Reply(text=req.text, ok=False)
    prepared = _generate_and_upload(svc, ctx, req)
    if isinstance(prepared, Reply):
        return Reply(text=prepared.text, ok=False)

    action = PendingAction(
        chat_id=ctx.chat_id, requester_id=ctx.user_id,
        intent=Intent.SUBMIT_SLURM, description=prepared.description,
        payload=prepared.payload,
    )
    ok, text = jobs.execute_submit(svc, ctx, action)
    return Reply(text=text, ok=ok)


def duplicate_note(svc: "BecarioService", owner_id: int, job_name: str, fingerprint: str) -> str:
    """Aviso si esto (o algo muy similar) ya se corrió: mismo job_name =
    mismo material y tipo de cálculo; misma huella = pedido idéntico."""
    if svc._calc_runs is None:
        return ""
    rows = svc._calc_runs.find_by_name(owner_id, job_name)
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
