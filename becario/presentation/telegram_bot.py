"""Capa de presentación: bot de Telegram.

Responsabilidad única: traducir updates de Telegram ↔ llamadas al
BecarioService. Acá no hay lógica de negocio ni construcción de comandos.

La autorización NO vive acá: se resuelve una sola vez, dentro del
servicio, contra el `UserRegistry`. Evita tener dos fuentes de verdad
sobre "quién puede usar el bot" (una allowlist acá y el roster allá).
"""
from __future__ import annotations

import asyncio
import html
import logging
import tempfile
from pathlib import Path
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..application.job_monitor import JobMonitorService
from ..application.services import FOREIGN_CONFIRMATION_TEXT, BecarioService, Reply
from ..domain.ports import ChatLogRepository, Transcriber

logger = logging.getLogger(__name__)


def _keyboard(token: str, allow_modify: bool = False) -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton("✅ Confirmar", callback_data=f"confirm:{token}"),
        InlineKeyboardButton("❌ Cancelar", callback_data=f"cancel:{token}"),
    ]
    if allow_modify:
        row.append(InlineKeyboardButton("✏️ Modificar", callback_data=f"modify:{token}"))
    return InlineKeyboardMarkup([row])


class TelegramBot:
    def __init__(
        self,
        token: str,
        service: BecarioService,
        job_monitor: Optional[JobMonitorService] = None,
        monitor_interval_seconds: float = 60.0,
        transcriber: Optional[Transcriber] = None,
        chat_log: Optional[ChatLogRepository] = None,
    ) -> None:
        self._service = service
        self._job_monitor = job_monitor
        self._transcriber = transcriber
        self._chat_log = chat_log
        self._app = Application.builder().token(token).build()
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self._app.add_handler(MessageHandler(filters.VOICE, self._on_voice))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        if job_monitor is not None:
            # job_queue requiere el extra python-telegram-bot[job-queue].
            self._app.job_queue.run_repeating(
                self._on_monitor_tick,
                interval=monitor_interval_seconds,
                first=monitor_interval_seconds,
            )

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Long polling: PTB maneja offset, reintentos y backoff solo."""
        logger.info("B.E.C.A.R.I.O. iniciando en modo polling…")
        self._app.run_polling(allowed_updates=["message", "callback_query"])

    # ------------------------------------------------------------------
    async def _log_chat(self, chat_id: Optional[int], role: str, text: str) -> None:
        """Registra un mensaje en la bitácora de conversación.

        La escritura a disco corre en un hilo aparte (`asyncio.to_thread`)
        para no bloquear el event loop con I/O sincrónica; el `await` en
        cada punto de llamada conserva el orden de los registros por chat.

        Nunca interrumpe al bot: si la persistencia falla (disco lleno,
        base bloqueada…), se loguea un warning y la conversación sigue."""
        if self._chat_log is None or chat_id is None:
            return
        try:
            await asyncio.to_thread(
                self._chat_log.add, chat_id=chat_id, role=role, text=text
            )
        except Exception as exc:
            logger.warning(
                "No pude registrar el mensaje en la bitácora (chat_id=%s): %s",
                chat_id, exc,
            )

    # ------------------------------------------------------------------
    async def _run_blocking(self, chat, fn, *args, **kwargs):
        """Ejecuta trabajo bloqueante (LLM, SSH, transcripción) en un hilo,
        manteniendo vivo el indicador «escribiendo…» del chat.

        Dos problemas de una vez: el usuario ve que el bot está procesando
        (una llamada al LLM en CPU puede superar los 30s y el silencio es
        indistinguible de un cuelgue), y el event loop queda libre para
        atender botones y el monitor mientras tanto — antes, el handler
        llamaba al servicio de forma síncrona y congelaba el loop entero.

        El servicio es thread-safe a propósito (locks en pending edits y
        confirmaciones); Telegram expira el chat action a los ~5s, por eso
        se reenvía. El indicador es cosmético: si falla, el trabajo sigue.
        """
        async def _keep_typing() -> None:
            while True:
                try:
                    await chat.send_chat_action(ChatAction.TYPING)
                except Exception:  # nunca interrumpir por el indicador
                    pass
                await asyncio.sleep(4.0)

        keeper = asyncio.create_task(_keep_typing()) if chat is not None else None
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        finally:
            if keeper is not None:
                keeper.cancel()

    # ------------------------------------------------------------------
    async def _send_reply(self, update: Update, reply: Reply) -> None:
        markup = (
            _keyboard(reply.confirmation_token, reply.allow_modify)
            if reply.needs_confirmation and reply.confirmation_token
            else None
        )
        if reply.monospace:
            # <pre> respeta el ancho fijo (tablas); el texto va escapado para
            # que ningún carácter del contenido se interprete como HTML.
            await update.effective_chat.send_message(
                f"<pre>{html.escape(reply.text)}</pre>",
                reply_markup=markup,
                parse_mode="HTML",
            )
        else:
            await update.effective_chat.send_message(reply.text, reply_markup=markup)
        # Se registra el texto crudo (sin el envoltorio <pre>): la bitácora
        # guarda contenido, no detalles de formato del canal.
        await self._log_chat(update.effective_chat.id, "bot", reply.text)

    # ------------------------------------------------------------------
    async def _on_text(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None:
            return
        text = update.message.text or ""
        await self._log_chat(update.effective_chat.id, "user", text)
        reply = await self._run_blocking(
            update.effective_chat,
            self._service.handle_text,
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            text=text,
        )
        await self._send_reply(update, reply)

    async def _on_voice(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None:
            return
        if self._transcriber is None:
            aviso = "🎙️ La transcripción de audio no está configurada."
            await update.effective_chat.send_message(aviso)
            await self._log_chat(update.effective_chat.id, "bot", aviso)
            return
        voice_file = await update.message.voice.get_file()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.ogg"
            await voice_file.download_to_drive(path)
            text = await self._run_blocking(
                update.effective_chat, self._transcriber.transcribe, path.read_bytes()
            )
        if not text.strip():
            aviso = "🎙️ No pude entender el audio, probá de nuevo."
            await update.effective_chat.send_message(aviso)
            await self._log_chat(update.effective_chat.id, "bot", aviso)
            return
        # La bitácora guarda la transcripción como mensaje del usuario:
        # es el texto que efectivamente entra al servicio.
        await self._log_chat(update.effective_chat.id, "user", text.strip())
        # Mostrar qué se entendió ANTES de actuar: si la transcripción vino
        # mal, el usuario lo ve enseguida y puede repetir.
        entendido = f"🎙️ Entendí: «{text.strip()}»"
        await update.effective_chat.send_message(entendido)
        await self._log_chat(update.effective_chat.id, "bot", entendido)
        reply = await self._run_blocking(
            update.effective_chat,
            self._service.handle_text,
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            text=text,
        )
        await self._send_reply(update, reply)

    async def _on_callback(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query.from_user is None:
            await query.answer("No autorizado.")
            return
        data = query.data or ""
        action, _, token = data.partition(":")
        # El toque de botón también es un pedido del usuario: la bitácora
        # guarda QUÉ se decidió, no solo la respuesta del bot.
        if query.message is not None:
            etiquetas = {"confirm": "confirmar", "cancel": "cancelar", "modify": "modificar"}
            await self._log_chat(query.message.chat.id, "user", etiquetas.get(action, data))
        # Confirmar ejecuta la acción real (sbatch/scancel por SSH): también
        # es trabajo bloqueante y merece indicador. `query.message` puede ser
        # None (mensaje viejo); en ese caso se corre sin typing.
        chat = query.message.chat if query.message is not None else None
        if action == "confirm":
            reply = await self._run_blocking(
                chat, self._service.confirm, token, requester_id=query.from_user.id
            )
        elif action == "cancel":
            reply = await self._run_blocking(
                chat, self._service.reject, token, requester_id=query.from_user.id
            )
        elif action == "modify":
            reply = await self._run_blocking(
                chat, self._service.start_modification, token,
                requester_id=query.from_user.id,
            )
        else:
            reply = Reply(text="⚠️ Acción desconocida.")
        if reply.text == FOREIGN_CONFIRMATION_TEXT:
            # Toque ajeno: avisar solo a quien tocó, sin ensuciar el chat ni
            # sacarle los botones a quien sí puede decidir.
            await query.answer(reply.text, show_alert=True)
            return
        await query.answer()
        # El mensaje original (con las condiciones del envío) se conserva
        # para poder revisarlo después: solo se quitan los botones, y el
        # resultado llega como mensaje aparte.
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:  # p. ej. mensaje demasiado viejo para editar
            logger.warning("No pude quitar los botones del mensaje original.")
        if query.message is not None:
            await query.message.chat.send_message(reply.text)
            await self._log_chat(query.message.chat.id, "bot", reply.text)

    # ------------------------------------------------------------------
    # Cierre del loop: aviso proactivo cuando un trabajo termina
    # ------------------------------------------------------------------
    async def _on_monitor_tick(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self._job_monitor is None:  # pragma: no cover - guard defensivo
            return
        for note in self._job_monitor.poll_and_notify():
            try:
                if note.monospace:
                    await context.bot.send_message(
                        chat_id=note.chat_id,
                        text=f"<pre>{html.escape(note.text)}</pre>",
                        parse_mode="HTML",
                    )
                else:
                    await context.bot.send_message(chat_id=note.chat_id, text=note.text)
                await self._log_chat(note.chat_id, "bot", note.text)
            except Exception as exc:
                logger.error("No pude notificar a chat_id=%s: %s", note.chat_id, exc)
