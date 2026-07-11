"""Capa de presentación: bot de Telegram.

Responsabilidad única: traducir updates de Telegram ↔ llamadas al
BecarioService. Acá no hay lógica de negocio ni construcción de comandos.

La autorización NO vive acá: se resuelve una sola vez, dentro del
servicio, contra el `UserRegistry`. Evita tener dos fuentes de verdad
sobre "quién puede usar el bot" (una allowlist acá y el roster allá).
"""
from __future__ import annotations

import html
import logging
import tempfile
from pathlib import Path
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..application.job_monitor import JobMonitorService
from ..application.services import BecarioService, Reply
from ..domain.ports import Transcriber

logger = logging.getLogger(__name__)


def _keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Confirmar", callback_data=f"confirm:{token}"),
                InlineKeyboardButton("❌ Cancelar", callback_data=f"cancel:{token}"),
            ]
        ]
    )


class TelegramBot:
    def __init__(
        self,
        token: str,
        service: BecarioService,
        job_monitor: Optional[JobMonitorService] = None,
        monitor_interval_seconds: float = 60.0,
        transcriber: Optional[Transcriber] = None,
    ) -> None:
        self._service = service
        self._job_monitor = job_monitor
        self._transcriber = transcriber
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
    async def _send_reply(self, update: Update, reply: Reply) -> None:
        markup = (
            _keyboard(reply.confirmation_token)
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
            return
        await update.effective_chat.send_message(reply.text, reply_markup=markup)

    # ------------------------------------------------------------------
    async def _on_text(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None:
            return
        reply = self._service.handle_text(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            text=update.message.text or "",
        )
        await self._send_reply(update, reply)

    async def _on_voice(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None:
            return
        if self._transcriber is None:
            await update.effective_chat.send_message(
                "🎙️ La transcripción de audio no está configurada."
            )
            return
        voice_file = await update.message.voice.get_file()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.ogg"
            await voice_file.download_to_drive(path)
            text = self._transcriber.transcribe(path.read_bytes())
        if not text.strip():
            await update.effective_chat.send_message(
                "🎙️ No pude entender el audio, probá de nuevo."
            )
            return
        reply = self._service.handle_text(
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
        if action == "confirm":
            reply = self._service.confirm(token, requester_id=query.from_user.id)
        elif action == "cancel":
            reply = self._service.reject(token, requester_id=query.from_user.id)
        else:
            reply = Reply(text="⚠️ Acción desconocida.")
        await query.answer()
        # Editar el mensaje original saca los botones y deja constancia.
        await query.edit_message_text(reply.text)

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
            except Exception as exc:
                logger.error("No pude notificar a chat_id=%s: %s", note.chat_id, exc)
