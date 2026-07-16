"""Handlers de consulta: historial de trabajos y resultados de corridas.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from pydantic import ValidationError

from ...domain.models import CalcKind, HistoryFilter
from ..context import Reply, _Ctx
from ..job_monitor import _parse_last_e0

if TYPE_CHECKING:
    from ..services import BecarioService


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


def query_history(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    try:
        flt = HistoryFilter(
            job_id=params.get("job_id"),
            name_contains=params.get("filtro_busqueda"),
            owner_id=ctx.user_id,  # nunca viene del LLM
        )
    except (ValidationError, ValueError) as exc:
        return Reply(text=f"⚠️ Filtro inválido:\n{exc}", ok=False)
    rows = svc._history.search(flt)
    if not rows:
        return Reply(text="📋 Historial:\nNo se encontraron registros.")
    return Reply(text="📋 Historial:\n" + _format_history_table(rows), monospace=True)


def query_results(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    if svc._calc_runs is None:
        return Reply(
            text="⚠️ La consulta de resultados no está configurada en este bot.",
            ok=False,
        )

    formula = params.get("formula") or params.get("formula_quimica")
    prefix = f"{str(formula).strip()}_" if formula else ""
    rows = svc._calc_runs.find_recent(ctx.user_id, prefix)
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
