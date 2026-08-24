"""Handlers de consulta: historial de trabajos y resultados de corridas.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from pydantic import ValidationError

from ...domain.datos_de_corrida import (
    DESCRIPCIONES,
    NINGUNO,
    DatoDeCorrida,
    contar_atomos,
    contar_pasos_ionicos,
    dato_valido,
    nsw_de,
    ultima_energia,
)
from ...domain.models import CalcKind, HistoryFilter
from ..context import Reply, _Ctx
from ..job_monitor import _parse_last_e0
from ..rechazos import registrar_rechazo, VOCABULARIO
# Los MISMOS topes de lectura que usa `relaxed_source` para el mismo par
# de archivos: si divergieran, dos caminos contestarían distinto sobre
# la convergencia de la misma corrida.
from ..relaxed_source import _INCAR_MAX_BYTES, _OSZICAR_MAX_BYTES

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

    # ¿Preguntaron por UN dato puntual? `dato` lo completa la segunda
    # pasada del router (`_backfill_dato`). Si no vino —plan viejo, router
    # caído, o pregunta general— se cae a la respuesta de siempre, que es
    # justo lo que `parametros_red` contesta.
    pedido = dato_valido(params.get("dato"))
    if pedido is not None and pedido is not DatoDeCorrida.PARAMETROS_RED:
        return _contestar_dato(ctx, pedido, row, run_dir)
    if str(params.get("dato") or "").strip().lower() == NINGUNO:
        # El modelo miró la pregunta y dijo que no está en el vocabulario.
        # Contestar los parámetros de red igual sería responder otra cosa
        # con cara de respuesta — el defecto que este camino vino a cerrar.
        registrar_rechazo(VOCABULARIO, user_id=ctx.user_id, detalle=run_dir)
        return _no_se_ese_dato()

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


def _no_se_ese_dato() -> Reply:
    """La pregunta cayó fuera del vocabulario: se dice, no se contesta otra.

    Es la misma decisión que el rebote de las heterostructuras: un «no sé»
    que además dice qué SÍ se puede es una respuesta útil; el dato de al
    lado con cara de correcto no lo es.
    """
    puedo = "\n".join(f"• {texto}" for texto in DESCRIPCIONES.values())
    return Reply(
        text=(
            "🤔 De una corrida terminada puedo sacarte estos datos, y ese no "
            f"está entre ellos:\n{puedo}\n\nSi el dato está en un archivo, "
            "pedímelo por nombre («mostrame el OSZICAR») y te lo muestro."
        ),
        ok=False,
    )


def _encabezado(row: dict) -> str:
    """De qué corrida se está hablando. Va en toda respuesta de dato: sin
    esto, un número suelto no dice sobre qué material salió."""
    return (
        f"{row.get('job_name', '?')} (job {row.get('job_id', '?')}, "
        f"{row.get('fecha', '?')})"
    )


def _contestar_dato(
    ctx: _Ctx, pedido: DatoDeCorrida, row: dict, run_dir: str
) -> Reply:
    """El dato puntual que se pidió, leído de los archivos de la corrida.

    Cada rama lee SOLO lo que necesita: preguntar cuántos átomos tiene la
    celda no tiene por qué bajar el OSZICAR entero.
    """
    cabeza = _encabezado(row)

    if pedido is DatoDeCorrida.PASOS_IONICOS:
        pasos = contar_pasos_ionicos(
            ctx.cluster.read_file(f"{run_dir}/OSZICAR", max_bytes=_OSZICAR_MAX_BYTES)
        )
        if pasos is None:
            return _sin_archivo("OSZICAR", run_dir)
        return Reply(text=f"🔄 {cabeza}: {pasos} paso(s) iónico(s).\n📂 {run_dir}")

    if pedido is DatoDeCorrida.ENERGIA:
        energia = ultima_energia(
            ctx.cluster.read_file(f"{run_dir}/OSZICAR", max_bytes=_OSZICAR_MAX_BYTES)
        )
        if energia is None:
            return _sin_archivo("OSZICAR", run_dir)
        return Reply(text=f"⚡ {cabeza}: E0 = {energia:.6f} eV\n📂 {run_dir}")

    if pedido is DatoDeCorrida.ATOMOS:
        texto = ctx.cluster.read_file(f"{run_dir}/CONTCAR") or ctx.cluster.read_file(
            f"{run_dir}/POSCAR"
        )
        atomos = contar_atomos(texto)
        if atomos is None:
            return _sin_archivo("CONTCAR/POSCAR", run_dir)
        return Reply(text=f"⚛️ {cabeza}: {atomos} átomo(s) en la celda.\n📂 {run_dir}")

    return _contestar_convergencia(ctx, cabeza, run_dir)


def _contestar_convergencia(ctx: _Ctx, cabeza: str, run_dir: str) -> Reply:
    """¿La relajación llegó al criterio, o se quedó sin pasos?

    Se compara el conteo del OSZICAR contra el NSW del INCAR, igual que
    `relaxed_source._check_convergence` — es el mismo criterio, y por la
    misma razón: la marca «reached required accuracy» está al FINAL del
    OUTCAR y `read_file` solo baja un prefijo.

    Un OSZICAR que llega al tope de lectura se declara NO VERIFICABLE en
    vez de contarse: sobre un prefijo faltan justo los pasos del final, así
    que el conteo daría de menos y diría «convergió». El error caro es ese.
    """
    oszicar = ctx.cluster.read_file(f"{run_dir}/OSZICAR", max_bytes=_OSZICAR_MAX_BYTES)
    incar = ctx.cluster.read_file(f"{run_dir}/INCAR", max_bytes=_INCAR_MAX_BYTES)
    pasos, nsw = contar_pasos_ionicos(oszicar), nsw_de(incar)
    if pasos is None or nsw is None:
        return Reply(
            text=f"⚠️ No pude verificar la convergencia de {cabeza}: falta el "
            f"OSZICAR o el NSW del INCAR.\n📂 {run_dir}",
            ok=False,
        )
    if oszicar is not None and len(oszicar) >= _OSZICAR_MAX_BYTES:
        return Reply(
            text=f"⚠️ El OSZICAR de {cabeza} es demasiado grande para "
            f"verificarlo desde acá; revisalo a mano.\n📂 {run_dir}",
            ok=False,
        )
    if nsw > 0 and pasos >= nsw:
        return Reply(
            text=f"⚠️ {cabeza} usó los {pasos} pasos iónicos que tenía "
            f"asignados (NSW={nsw}) sin alcanzar el criterio de fuerzas: esa "
            f"estructura NO está relajada del todo.\n📂 {run_dir}",
            ok=False,
        )
    return Reply(
        text=f"✅ {cabeza} convergió: {pasos} de {nsw} pasos iónicos "
        f"disponibles.\n📂 {run_dir}"
    )


def _sin_archivo(nombre: str, run_dir: str) -> Reply:
    return Reply(
        text=f"⚠️ No pude leer el {nombre} de esa corrida.\n📂 {run_dir}\n"
        "¿Sigue existiendo en el cluster?",
        ok=False,
    )


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
    # Los botones también son «cómo estoy hecho», y son la parte que el
    # usuario mira cuando pregunta por la interfaz. Frente a un batch de
    # ocho pasos equivocado, alguien escribió «acá debería haber una opción
    # de modificar» y recibió los tres bullets de arriba: ciertos, y sobre
    # otra cosa. El ✏️ estaba a un campo de distancia de existir.
    lines.append(
        "• Botones: cuando te muestro un plan, nada se ejecuta hasta que "
        "toques ✅. Con ✏️ lo corregís sin reescribirlo entero (podés "
        "apuntar el paso: «paso 2: 4 nodos») y con ❌ lo descartás."
    )

    return Reply(text="\n".join(lines))
