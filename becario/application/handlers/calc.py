"""Handlers de estructuras y cálculos VASP: genera inputs, sube y prepara.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ...domain.models import (
    CalcKind,
    Intent,
    OutputFormat,
    PendingAction,
    PendingPlan,
    SlurmJobRequest,
    StructureKind,
    StructureRequest,
    VaspCalcRequest,
)
from ..context import Reply, _Ctx

if TYPE_CHECKING:
    from ..services import BecarioService


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


def prepare_calc(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
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
        result = svc._calc_inputs.generate(req)
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
            job_name=f"{req.formula}_{calc_kind.value}",
            partition=req.partition,
            nodes=req.nodes,
            time_limit=req.time_limit,
            script_path=f"{remote_run_dir}/run_vasp.sh",
        )
    except (ValidationError, ValueError) as exc:
        return Reply(text=f"⚠️ Parámetros inválidos para el envío:\n{exc}")

    fingerprint = _calc_fingerprint(req)
    duplicate = duplicate_note(svc, ctx.user_id, slurm_req.job_name, fingerprint)

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
    plan = PendingPlan(
        chat_id=ctx.chat_id, requester_id=ctx.user_id, steps=[action],
        decision_id=ctx.decision_id,
    )
    token = svc._confirmations.put(plan)
    # El aviso de duplicado va PRIMERO: al final de un mensaje largo
    # pasa desapercibido.
    warning_block = f"{duplicate}\n\n" if duplicate else ""
    return Reply(
        text=(
            f"{warning_block}✅ Inputs generados y subidos al cluster.\n\n"
            f"⚠️ ¿Confirmás el envío?\n\n{action.description}"
        ),
        needs_confirmation=True,
        confirmation_token=token,
        allow_modify=True,
    )


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
