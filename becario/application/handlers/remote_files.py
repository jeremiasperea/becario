"""Handlers de archivos remotos: crear carpetas, listar y ver archivos.

Funciones de módulo extraídas de `BecarioService`: reciben la fachada como
primer argumento `svc` y conservan el comportamiento original sin cambios.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ...domain.models import ListFilesRequest, RemoteDirRequest, ViewFileRequest
from ..context import Reply, _Ctx

if TYPE_CHECKING:
    from ..services import BecarioService

logger = logging.getLogger(__name__)

_LISTING_MAX_CHARS = 3500  # margen bajo el límite de 4096 de Telegram
# Tope de lectura para `ver_archivo`: el nombre lo elige el usuario, así que
# no podemos bajar entero un archivo arbitrario del run (WAVECAR/CHGCAR pesan
# GB). 64 KiB sobra para llenar el recorte de Telegram y acota la memoria.
_VIEW_FILE_MAX_BYTES = 64 * 1024


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


def _truncate_file_content(text: str) -> str:
    """Recorta el contenido de un archivo largo (en un borde de línea si
    se puede) para que la respuesta entre en un mensaje de Telegram."""
    if len(text) <= _LISTING_MAX_CHARS:
        return text
    cut = text.rfind("\n", 0, _LISTING_MAX_CHARS)
    if cut <= 0:
        cut = _LISTING_MAX_CHARS
    return text[:cut].rstrip() + "\n… (archivo truncado)"


_HOME_NO_RESUELTO = Reply(
    text="⚠️ No pude resolver el home remoto. Revisá la conexión al cluster.",
    ok=False,
)


def _corridas_dir(svc: "BecarioService", ctx: _Ctx) -> str | None:
    """El `remote_base` como ruta absoluta (anclado al home si es relativo)."""
    base = svc._remote_base
    if base.startswith("/"):
        return base
    home = ctx.cluster.home_dir()
    return f"{home}/{base}" if home else None


def _resolve_workspace_path(
    svc: "BecarioService",
    ctx: _Ctx,
    raw: str,
    base: str | None = None,
    solo_workspace: bool = False,
) -> tuple[str | None, Reply | None]:
    """Resuelve una ruta del usuario contra el ancla que declaró el router.

    `base` es lo que el LLM eligió: 'home', 'corridas' o 'absoluta'. Existe
    porque antes había solo dos formas —absoluta o relativa a las corridas—
    y el home del usuario NO era representable: para decir «mi home» el
    modelo tenía que inventar una ruta literal, y la inventaba
    (`/home/ana`). No era un problema del modelo sino del vocabulario que
    le dábamos.

    Sin `base` (planes viejos, o un modelo que no lo completó) se cae al
    criterio de antes: absoluta si empieza con '/', relativa a las
    corridas si no. Eso mantiene andando todo lo que ya funcionaba.

    `solo_workspace` acota las rutas absolutas al home o a las corridas.
    Lo pide ESCRIBIR, no leer: así se crearon `/Zr/bcc` y compañía en la
    raíz del cluster, invisibles para el listado que el usuario pedía
    después. Leer una ruta compartida (`/data/otro_grupo`) es legítimo y
    de eso se ocupan los permisos del cluster, que es donde vive el
    aislamiento según ADR-0004 — no la lógica de la aplicación.
    """
    raw = str(raw or "").strip()
    ancla = (base or "").strip().lower()

    if ancla == "home":
        home = ctx.cluster.home_dir()
        if not home:
            return None, _HOME_NO_RESUELTO
        return (f"{home}/{raw}" if raw else home), None

    if ancla == "absoluta" or (not ancla and raw.startswith("/")):
        if not raw.startswith("/"):
            # El LLM dijo 'absoluta' pero mandó algo relativo: manda la
            # forma real de la ruta, no la etiqueta — antes que fabricar
            # una barra que el usuario nunca escribió.
            return _resolve_workspace_path(
                svc, ctx, raw, base="corridas", solo_workspace=solo_workspace
            )
        return _validar_absoluta(svc, ctx, raw, solo_workspace)

    # 'corridas' y cualquier cosa que no reconozcamos: el default seguro.
    if raw.startswith("/"):
        # Ancla 'corridas' con una ruta absoluta pegada: se trata como
        # absoluta en vez de concatenar y producir '/corridas//Zr'.
        return _validar_absoluta(svc, ctx, raw, solo_workspace)
    corridas = _corridas_dir(svc, ctx)
    if corridas is None:
        return None, _HOME_NO_RESUELTO
    return (f"{corridas}/{raw}" if raw else corridas), None


def _validar_absoluta(
    svc: "BecarioService", ctx: _Ctx, path: str, solo_workspace: bool
) -> tuple[str | None, Reply | None]:
    """Una ruta absoluta pasa tal cual.

    Se evaluó acotarlas al home y a las corridas —era el punto por donde
    entraron `/Zr/bcc` y compañía a la raíz del cluster— y se descartó:
    bloqueaba pedidos legítimos («creá /data/proyectos/x» en un área
    compartida) para tapar un síntoma cuya causa era otra. Esas rutas no
    las escribió el usuario, las INVENTÓ el modelo porque «mi home» y «la
    carpeta de corridas» no eran expresables; con `base` en el schema ya
    no tiene que inventarlas (medido: 30/30). El aislamiento sigue donde
    dice ADR-0004: en los permisos del cluster, no acá.

    `solo_workspace` queda como parámetro porque la decisión puede
    revisarse con datos —si la batería vuelve a mostrar rutas inventadas,
    este es el lugar—, pero hoy nadie la activa.
    """
    return (path.rstrip("/") or "/"), None


def create_directory(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    raw = params.get("destino_remoto")
    if not raw:
        return Reply(
            text="⚠️ Decime la carpeta que querés crear, p. ej.: "
            '"creame la carpeta Zr/bcc" (relativa a tus corridas) '
            "o una ruta absoluta.",
            ok=False,
        )
    path, err = _resolve_workspace_path(
        svc, ctx, raw, params.get("base"), solo_workspace=True
    )
    if err is not None:
        return err
    try:
        req = RemoteDirRequest(path=path)
    except (ValidationError, ValueError):
        return Reply(
            text=f"⚠️ Ruta inválida: {raw!r}. Puede ser relativa a tus "
            "corridas (Zr/bcc) o absoluta (/data/x), sin caracteres "
            "especiales ni '..'.",
            ok=False,
        )
    result = ctx.cluster.make_directory(req.path)
    if not result.ok:
        # Sin esto, un mkdir fallido (permisos, cuota) solo se ve en el chat.
        logger.warning("mkdir remoto falló para %s: %s", req.path, result.message)
    status = "✅" if result.ok else "❌"
    return Reply(text=f"{status} 📁 {result.message}", ok=result.ok)


def list_files(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    # `base` dice contra qué se resuelve ('mi home' vs 'mis corridas'); sin
    # él se cae al default de siempre, la base de corridas.
    raw = params.get("destino_remoto") or ""
    path, err = _resolve_workspace_path(svc, ctx, raw, params.get("base"))
    if err is not None:
        return err
    try:
        req = ListFilesRequest(path=path)
    except (ValidationError, ValueError):
        return Reply(
            text=f"⚠️ Ruta inválida: {raw!r}. Puede ser relativa a tus "
            "corridas (Zr/bcc) o absoluta (/data/x), sin caracteres "
            "especiales ni '..'.",
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


def view_file(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    """Muestra el contenido de un archivo remoto (solo lectura).

    El archivo se identifica por ruta absoluta (`destino_remoto`) o por
    nombre suelto (`nombre_archivo`) resuelto contra el directorio de la
    última corrida del usuario — esto último arregla el pedido real
    "ver el CONTCAR", que antes el LLM resolvía inventando un `ls`.
    """
    path = params.get("destino_remoto")
    filename = params.get("nombre_archivo")
    if not (path or filename):
        return Reply(
            text="⚠️ Decime qué archivo querés ver: un nombre "
            '(p. ej. "mostrame el CONTCAR") o una ruta absoluta.',
            ok=False,
        )
    if path and filename:
        return Reply(
            text="⚠️ Me diste un nombre y una ruta a la vez. Pasame solo "
            "uno: el nombre del archivo o la ruta absoluta.",
            ok=False,
        )
    try:
        req = ViewFileRequest(path=path, filename=filename)
    except (ValidationError, ValueError):
        return Reply(
            text="⚠️ Pedido de archivo inválido: el nombre no puede tener "
            "'/' ni '..', y una ruta tiene que ser absoluta (empezar con /).",
            ok=False,
        )
    if req.path:
        remote_path = req.path
    else:
        if svc._calc_runs is None:
            return Reply(
                text="⚠️ Ver archivos por nombre necesita el historial de "
                "corridas, que no está configurado en este bot. Pasame la "
                "ruta absoluta del archivo.",
                ok=False,
            )
        formula = params.get("formula") or params.get("formula_quimica")
        prefix = f"{str(formula).strip()}_" if formula else ""
        rows = svc._calc_runs.find_recent(ctx.user_id, prefix)
        if not rows:
            de = f" de {formula}" if formula else ""
            return Reply(
                text=f"📭 No encontré corridas tuyas{de} para resolver "
                f"{req.filename!r}. Pasame la ruta absoluta del archivo.",
                ok=False,
            )
        run_dir = str(rows[0].get("run_dir", "")).rstrip("/")
        if not run_dir:
            return Reply(
                text="⚠️ La corrida más reciente no tiene directorio asociado.",
                ok=False,
            )
        remote_path = f"{run_dir}/{req.filename}"
    content = ctx.cluster.read_file(remote_path, max_bytes=_VIEW_FILE_MAX_BYTES)
    if content is None:
        return Reply(
            text=f"❌ 📄 No pude leer {remote_path}. ¿Existe en el cluster?",
            ok=False,
        )
    if not content.strip():
        return Reply(text=f"📄 {remote_path}:\n(archivo vacío)")
    return Reply(
        text=f"📄 {remote_path}:\n{_truncate_file_content(content)}",
        monospace=True,
    )
