"""Handlers de consulta: historial de trabajos y resultados de corridas.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
"""
from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Optional

from pydantic import ValidationError

from ...domain.models import CalcKind, HistoryFilter
from ...domain.sugerencias import CorridaPrevia, render
from ..context import Reply, _Ctx
from ..job_monitor import _parse_last_e0

if TYPE_CHECKING:
    from ..services import BecarioService

logger = logging.getLogger(__name__)


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


def _corridas_previas(
    svc: "BecarioService", owner_id: int, formula: str
) -> list[CorridaPrevia]:
    """Traduce lo que hay en la base a lo que las reglas necesitan.

    La huella de cada corrida guarda los parámetros con los que se pidió
    (`calc_kind`, `encut`, …) y el historial guarda cómo terminó. Ninguna
    de las dos alcanza sola: la huella no sabe si el trabajo falló, y el
    historial no sabe con qué ENCUT se corrió.
    """
    filas = svc._calc_runs.find_recent(owner_id, f"{formula}_", limit=20)
    estados = {}
    if svc._history is not None:
        # El historial tiene una fila por transición ('enviado', y después
        # el desenlace), así que la ÚLTIMA de cada job es la que vale.
        try:
            # `limit` está topeado en 50 por el dominio, y el nombre importa:
            # un `limite=` mal escrito lo ignora pydantic en silencio y deja
            # el default de 5 — cinco filas de historial no alcanzan para
            # cubrir las corridas de un material trabajado.
            for h in svc._history.search(HistoryFilter(owner_id=owner_id, limit=50)):
                # `search` devuelve `ORDER BY fecha DESC`, así que la PRIMERA
                # fila de cada trabajo es su estado más reciente y las
                # siguientes son su pasado. Con un `=` en vez de
                # `setdefault` ganaba la más vieja: un trabajo que quedó en
                # 'falló' volvía como 'enviado' y la regla más importante
                # —no encadenar sobre una corrida rota— no disparaba nunca,
                # sin que nada avisara.
                estados.setdefault(str(h.get("job_id")), str(h.get("estado") or ""))
        except Exception:  # el historial es contexto, no puede tirar la consulta
            logger.warning("no pude leer el historial para las sugerencias", exc_info=True)

    corridas = []
    for f in filas:
        try:
            huella = json.loads(f.get("fingerprint") or "{}")
            kind = CalcKind(huella.get("calc_kind"))
        except (ValueError, TypeError):
            continue  # una huella vieja o rota no vale una excepción
        corridas.append(CorridaPrevia(
            formula=str(huella.get("formula") or formula),
            calc_kind=kind,
            fecha=str(f.get("fecha") or ""),
            estado=estados.get(str(f.get("job_id")), ""),
            encut=huella.get("encut"),
            job_name=str(f.get("job_name") or ""),
        ))
    return corridas


def suggest(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    """Qué le falta a un material, mirando lo que ya se corrió.

    El LLM decide que esto es un pedido de sugerencia; QUÉ sugerir lo
    decide el dominio (`sugerencias.py`), con reglas y citando el manual.
    Un modelo de 7B opinando sobre metodología DFT es exactamente la clase
    de respuesta creíble y equivocada que el resto del proyecto evita.

    No toca el cluster: se contesta con lo que ya está registrado.
    """
    if svc._calc_runs is None:
        return Reply(
            text="⚠️ El registro de corridas no está configurado en este bot, "
            "así que no puedo mirar qué corriste antes.", ok=False,
        )

    formula = str(params.get("formula") or "").strip()
    if not formula:
        return Reply(
            text="🧭 Decime de qué material, p. ej.: «¿qué me falta para el Zr?».",
            ok=False, awaiting_params=True,
        )

    corridas = _corridas_previas(svc, ctx.user_id, formula)
    return Reply(text=render(formula, corridas), ok=bool(corridas))


def explain(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    """Contesta una pregunta sobre el bot mismo, SIN tocar el cluster.

    Nace de un caso concreto: el bot dijo «No encontré POTCAR para O en
    /data/potcars (busqué O_sv, O_pv, O)», el usuario preguntó «donde
    buscaste?» y recibió la cola de trabajos. Como toda intención mapeaba
    a una acción del cluster, una pregunta se ruteaba a la acción más
    parecida — y la respuesta ya estaba en el mensaje anterior del propio
    bot.

    Responde con lo que el servicio YA tiene en la mano: dónde busca los
    pseudopotenciales y en qué orden, dónde deja las corridas y con qué
    cuenta entra. Nada de esto necesita red, así que tampoco puede fallar
    por el cluster.

    LIMITACIÓN conocida: mientras hay un pedido esperando respuesta,
    `handle_text` deriva a `_apply_edit` y no llega a rutear, así que esta
    intención es inalcanzable — preguntar «¿qué estabas esperando?» se
    interpreta como el dato que falta. El caso de la bitácora no cae ahí
    (la pregunta vino después de un error, sin pendiente vivo), pero la
    puerta queda abierta: haría falta que `_apply_edit` reconozca una
    pregunta como hoy reconoce «cancelar».
    """
    lines = ["🛠️ Así estoy configurado:"]

    if svc._potcar_dir:
        variantes = ", ".join(
            f"<elemento>{v}" if v else "<elemento>" for v in svc._POTCAR_VARIANTS
        )
        lines.append(
            f"• POTCAR: los busco en {svc._potcar_dir}, probando en este "
            f"orden {variantes} (las variantes semi-core primero, como "
            f"recomienda VASP para casi todos los metales de transición)."
        )
    else:
        lines.append(
            "• POTCAR: no tengo configurada la biblioteca de "
            "pseudopotenciales (BECARIO_POTCAR_DIR), así que no puedo "
            "preparar cálculos."
        )

    lines.append(f"• Corridas: las dejo en {svc._remote_base} de tu cuenta del cluster.")
    lines.append(f"• Cuenta: entro como {ctx.identity.ssh_user}.")

    return Reply(text="\n".join(lines))
