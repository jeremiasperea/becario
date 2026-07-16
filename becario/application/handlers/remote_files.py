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


def create_directory(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    raw = params.get("destino_remoto")
    if not raw:
        return Reply(
            text="⚠️ Decime la ruta de la carpeta que querés crear, "
            'p. ej.: "creame la carpeta /home/usuario/pruebas".',
            ok=False,
        )
    try:
        req = RemoteDirRequest(path=str(raw))
    except (ValidationError, ValueError):
        return Reply(
            text=f"⚠️ Ruta inválida: {raw!r}. Tiene que ser una ruta "
            "absoluta (empezar con /), sin caracteres especiales ni '..'.",
            ok=False,
        )
    result = ctx.cluster.make_directory(req.path)
    if not result.ok:
        # Sin esto, un mkdir fallido (permisos, cuota) solo se ve en el chat.
        logger.warning("mkdir remoto falló para %s: %s", req.path, result.message)
    status = "✅" if result.ok else "❌"
    return Reply(text=f"{status} 📁 {result.message}", ok=result.ok)


def list_files(svc: "BecarioService", ctx: _Ctx, params: dict) -> Reply:
    raw = params.get("destino_remoto")
    if not raw:
        # Sin ruta: listar la base de corridas, resuelta contra el home
        # remoto si la config es relativa (mismo criterio que al subir
        # una corrida).
        base = svc._remote_base
        if not base.startswith("/"):
            home = ctx.cluster.home_dir()
            if not home:
                return Reply(
                    text="⚠️ No pude resolver el home remoto para listar. "
                    "Revisá la conexión al cluster.",
                    ok=False,
                )
            base = f"{home}/{base}"
        raw = base
    try:
        req = ListFilesRequest(path=str(raw))
    except (ValidationError, ValueError):
        return Reply(
            text=f"⚠️ Ruta inválida: {raw!r}. Tiene que ser una ruta "
            "absoluta (empezar con /), sin caracteres especiales ni '..'.",
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
