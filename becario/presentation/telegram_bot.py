"""Capa de presentación: bot de Telegram.

Responsabilidad única: traducir updates de Telegram ↔ llamadas al
BecarioService. Acá no hay lógica de negocio ni construcción de comandos.

La autorización NO vive acá: se resuelve una sola vez, dentro del
servicio, contra el `UserRegistry`. Evita tener dos fuentes de verdad
sobre "quién puede usar el bot" (una allowlist acá y el roster allá).
"""
from __future__ import annotations

import asyncio
import functools
import html
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

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
from ..incidentes import FALLO_INESPERADO, FALLO_TRAS_CONFIRMAR, nuevo_incidente

logger = logging.getLogger(__name__)

# Telegram rechaza los mensajes de más de 4096 caracteres. El margen cubre el
# envoltorio `<pre>…</pre>` y cualquier redondeo del conteo.
_LIMITE_TELEGRAM = 4096
_PRESUPUESTO = _LIMITE_TELEGRAM - 96


def _corte_maximo(linea: str, medir, presupuesto: int) -> int:
    """Prefijo más largo de `linea` que entra en el presupuesto, por búsqueda
    binaria. Hace falta porque `medir` no es la identidad: con `html.escape`
    un `&` ocupa cinco caracteres, así que la longitud cruda no sirve para
    saber dónde cortar."""
    bajo, alto, mejor = 1, min(len(linea), presupuesto), 1
    while bajo <= alto:
        medio = (bajo + alto) // 2
        if medir(linea[:medio]) <= presupuesto:
            mejor, bajo = medio, medio + 1
        else:
            alto = medio - 1
    return mejor


def _trocear(text: str, medir=len, presupuesto: int = _PRESUPUESTO) -> list[str]:
    """Parte `text` en trozos que entren en un mensaje de Telegram.

    Corta por líneas siempre que puede: partir al medio una tabla de
    convergencia o la cola de un log la vuelve ilegible. Solo cuando una
    línea sola no entra se la parte a lo bruto.

    Vive acá, en el borde, y no en cada handler a propósito. El límite es de
    Telegram, así que un guard por handler protege los casos que alguien se
    acordó de proteger — y el reporte de un plan de siete pasos, el
    diagnóstico de un fallo con dos colas de log y la tabla del barrido de
    ENCUT viajaban sin nada.
    """
    if medir(text) <= presupuesto:
        return [text]
    trozos: list[str] = []
    actual = ""
    for linea in text.split("\n"):
        while medir(linea) > presupuesto:
            if actual:
                trozos.append(actual)
                actual = ""
            corte = _corte_maximo(linea, medir, presupuesto)
            trozos.append(linea[:corte])
            linea = linea[corte:]
        candidato = f"{actual}\n{linea}" if actual else linea
        if actual and medir(candidato) > presupuesto:
            trozos.append(actual)
            actual = linea
        else:
            actual = candidato
    if actual:
        trozos.append(actual)
    return trozos


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
        max_workers: int = 8,
        al_cerrar: Optional[Callable[[], None]] = None,
    ) -> None:
        self._service = service
        self._job_monitor = job_monitor
        self._transcriber = transcriber
        self._chat_log = chat_log
        # Qué hacer al terminar (cerrar las conexiones SSH, típicamente).
        # Se inyecta un callable y no el factory para que la presentación
        # siga sin conocer la infraestructura.
        self._al_cerrar = al_cerrar
        # Pool PROPIO para el trabajo bloqueante (LLM, SSH, transcripción),
        # en vez del executor default de asyncio.
        #
        # No es afinar: es aislar. El default lo comparte todo el proceso,
        # así que un hilo trabado ahí se lleva capacidad de cualquier otra
        # cosa que use `to_thread` — la bitácora, por ejemplo. Con un pool
        # nombrado, la saturación se ve en los nombres de los hilos de un
        # `py-spy`/`faulthandler` en vez de tener que adivinarla.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="becario-blocking"
        )
        self._app = Application.builder().token(token).build()
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self._app.add_handler(MessageHandler(filters.VOICE, self._on_voice))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        self._app.add_error_handler(self._on_error)
        if job_monitor is not None:
            # job_queue requiere el extra python-telegram-bot[job-queue].
            self._app.job_queue.run_repeating(
                self._on_monitor_tick,
                interval=monitor_interval_seconds,
                first=monitor_interval_seconds,
            )

    # ------------------------------------------------------------------
    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Red de contención: ninguna excepción termina en silencio.

        Sin esto, cualquier fallo no previsto —una base bloqueada, un
        `BadRequest` al enviar, un reventón de una biblioteca de terceros—
        subía a python-telegram-bot, se logueaba y ahí moría. El usuario se
        quedaba mirando un chat quieto, sin siquiera el «escribiendo…».

        No es una hipótesis: en la bitácora hay dos pedidos (mensajes 23 y
        88) donde `decisiones_router` registra que el servicio ruteó y
        respondió, y al chat no llegó nada. El del mensaje 88 esperó casi
        once minutos antes de repetirlo.
        """
        incidente = nuevo_incidente()
        logger.error(
            "Incidente %s: excepción no atrapada procesando un update",
            incidente, exc_info=context.error,
        )
        chat = getattr(update, "effective_chat", None)
        if chat is None:
            # Errores sin chat asociado (p. ej. del propio job_queue): queda
            # el log, que es todo lo que se puede hacer.
            return
        aviso = FALLO_INESPERADO.format(incidente=incidente)
        try:
            await self._enviar(chat.send_message, aviso, chat_id=chat.id)
        except Exception:
            # Si tampoco se puede avisar, el log es el último recurso. Este
            # except NO puede faltar: una excepción acá vuelve al mismo
            # handler y arma un bucle.
            logger.error("Incidente %s: tampoco pude avisarle al usuario.", incidente)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Long polling: PTB maneja offset, reintentos y backoff solo."""
        logger.info("B.E.C.A.R.I.O. iniciando en modo polling…")
        try:
            # PTB ya atiende SIGINT/SIGTERM y devuelve el control acá, así
            # que el `finally` corre también en un `systemctl stop`. No hace
            # falta instalar handlers propios (y pelearlos con los suyos).
            self._app.run_polling(allowed_updates=["message", "callback_query"])
        finally:
            # `wait=False`: al salir puede haber un hilo esperando al cluster,
            # y no tiene sentido demorar el apagado por él. Los hilos son
            # daemon, así que no impiden que el proceso termine.
            self._executor.shutdown(wait=False)
            if self._al_cerrar is not None:
                try:
                    self._al_cerrar()
                except Exception:
                    # Un apagado que revienta no puede tapar el motivo real
                    # de la salida.
                    logger.exception("Falló el cierre ordenado")
            logger.info("B.E.C.A.R.I.O. terminó.")

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
    async def _en_hilo(self, fn, *args, **kwargs):
        """Corre `fn` en el pool propio del bot.

        Equivalente a `asyncio.to_thread` salvo por el executor, que es lo
        único que interesa: el default es del proceso entero.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(fn, *args, **kwargs)
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
            return await self._en_hilo(fn, *args, **kwargs)
        finally:
            if keeper is not None:
                keeper.cancel()

    # ------------------------------------------------------------------
    async def _enviar(
        self,
        enviar,
        text: str,
        *,
        chat_id: Optional[int],
        markup=None,
        monospace: bool = False,
    ) -> None:
        """Único punto de salida de texto hacia Telegram.

        Trocea si hace falta y registra en la bitácora el texto ENTERO una
        sola vez: la bitácora guarda contenido, no en cuántos mensajes lo
        partió el canal (ni el envoltorio `<pre>`).
        """
        if monospace:
            # <pre> respeta el ancho fijo (tablas); el texto va escapado para
            # que ningún carácter del contenido se interprete como HTML.
            def medir(t: str) -> int:
                return len(html.escape(t)) + len("<pre></pre>")
        else:
            medir = len

        trozos = _trocear(text, medir)
        for i, trozo in enumerate(trozos):
            cuerpo = f"<pre>{html.escape(trozo)}</pre>" if monospace else trozo
            kwargs = {"parse_mode": "HTML"} if monospace else {}
            # Los botones van en el ÚLTIMO trozo: colgados del primero
            # quedarían arriba del texto que hay que leer para decidir.
            if markup is not None and i == len(trozos) - 1:
                kwargs["reply_markup"] = markup
            await enviar(cuerpo, **kwargs)
        await self._log_chat(chat_id, "bot", text)

    async def _send_reply(self, update: Update, reply: Reply) -> None:
        markup = (
            _keyboard(reply.confirmation_token, reply.allow_modify)
            if reply.needs_confirmation and reply.confirmation_token
            else None
        )
        chat = update.effective_chat
        await self._enviar(
            chat.send_message,
            reply.text,
            chat_id=chat.id,
            markup=markup,
            monospace=reply.monospace,
        )
        # Confirmaciones individuales de un plan (una por cálculo), cada
        # una con sus propios botones.
        for followup in reply.followups:
            await self._send_reply(update, followup)

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
        chat = update.effective_chat
        if self._transcriber is None:
            await self._enviar(
                chat.send_message,
                "🎙️ La transcripción de audio no está configurada.",
                chat_id=chat.id,
            )
            return
        voice_file = await update.message.voice.get_file()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.ogg"
            await voice_file.download_to_drive(path)
            text = await self._run_blocking(
                update.effective_chat, self._transcriber.transcribe, path.read_bytes()
            )
        if not text.strip():
            await self._enviar(
                chat.send_message,
                "🎙️ No pude entender el audio, probá de nuevo.",
                chat_id=chat.id,
            )
            return
        # La bitácora guarda la transcripción como mensaje del usuario:
        # es el texto que efectivamente entra al servicio.
        await self._log_chat(chat.id, "user", text.strip())
        # Mostrar qué se entendió ANTES de actuar: si la transcripción vino
        # mal, el usuario lo ve enseguida y puede repetir.
        await self._enviar(
            chat.send_message, f"🎙️ Entendí: «{text.strip()}»", chat_id=chat.id
        )
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
            # `chat` acá es un `Chat`, y `start_modification` espera el ID:
            # ese valor termina en `_PendingEdit.chat_id` y de ahí en el
            # `send_message` del barrido de vencidos, que con un objeto en
            # vez de un entero fallaba y se comía el aviso (el `except` de
            # abajo lo tapaba). Sin `message` no hay a quién avisarle: 0 es
            # el centinela que `sweep_expired_pendings` ya interpreta como
            # "vencer callado".
            reply = await self._run_blocking(
                chat, self._service.start_modification, token,
                requester_id=query.from_user.id,
                chat_id=chat.id if chat is not None else 0,
            )
        else:
            reply = Reply(text="⚠️ Acción desconocida.")
        if reply.text == FOREIGN_CONFIRMATION_TEXT:
            # Toque ajeno: avisar solo a quien tocó, sin ensuciar el chat ni
            # sacarle los botones a quien sí puede decidir.
            await self._acusar_toque(query, reply.text, alerta=True)
            return
        await self._acusar_toque(query)
        # El mensaje original (con las condiciones del envío) se conserva
        # para poder revisarlo después: solo se quitan los botones, y el
        # resultado llega como mensaje aparte.
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:  # p. ej. mensaje demasiado viejo para editar
            logger.warning("No pude quitar los botones del mensaje original.")
        if query.message is None:
            return
        try:
            await self._enviar(
                query.message.chat.send_message,
                reply.text,
                chat_id=query.message.chat.id,
                monospace=reply.monospace,
            )
        except Exception:
            if action != "confirm":
                raise  # nada se ejecutó: el handler global dice la verdad
            await self._fallo_entregando_tras_confirmar(query.message.chat)

    async def _acusar_toque(
        self, query, texto: Optional[str] = None, *, alerta: bool = False
    ) -> None:
        """Acusa recibo del toque, y NO puede tumbar el resto.

        Telegram caduca los callbacks: `answer()` sobre uno viejo levanta
        `BadRequest("Query is too old and response timeout expired")`. Pasó
        dos veces en una sola sesión, sobre una tarjeta de quince minutos, y
        el usuario recibió un incidente —que suena a que el bot está roto—
        cuando lo único que había pasado es que se tardó. Peor: la
        confirmación ya había vencido por su propio TTL, así que `reject`
        tenía listo un «⌛ Esta confirmación expiró o ya fue usada» que
        explica exactamente qué pasó, y ese mensaje nunca llegó a mandarse.

        La línea de abajo —`edit_message_reply_markup`— ya contemplaba esta
        misma condición. Quien la escribió previó el mensaje viejo para el
        `edit` y no para el `answer`, una línea antes.
        """
        try:
            if texto is None:
                await query.answer()
            else:
                await query.answer(texto, show_alert=alerta)
        except Exception:
            logger.warning(
                "No pude acusar el toque del botón (¿callback vencido?).",
                exc_info=True,
            )

    async def _fallo_entregando_tras_confirmar(self, chat) -> None:
        """No se pudo entregar el resultado de un ✅. El aviso NO puede decir
        «no ejecuté nada».

        Para cuando esto corre, `confirm` ya volvió: el `sbatch` pudo haber
        salido. El texto del handler global dice «no ejecuté nada, podés
        volver a intentarlo», y sobre un trabajo ya encolado eso son dos
        trabajos en la cola y dos veces las horas de cómputo por un error
        que el usuario nunca vio.

        Desde acá no hay forma de saber si se ejecutó, así que se dice eso.
        Se falla hacia el lado barato: mandar a verificar de más molesta,
        afirmar que no pasó nada cuando pasó cuesta plata.
        """
        incidente = nuevo_incidente()
        logger.exception(
            "Incidente %s: no pude entregar el resultado de una confirmación",
            incidente,
        )
        try:
            await self._enviar(
                chat.send_message,
                FALLO_TRAS_CONFIRMAR.format(incidente=incidente),
                chat_id=chat.id,
            )
        except Exception:
            logger.error("Incidente %s: tampoco pude avisarle al usuario.", incidente)

    # ------------------------------------------------------------------
    # Cierre del loop: aviso proactivo cuando un trabajo termina
    # ------------------------------------------------------------------
    async def _on_monitor_tick(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Las dos consultas van A UN HILO, no al event loop.
        #
        # `poll_and_notify()` hace varias idas y vueltas SSH por cada trabajo
        # activo, y hasta acá corría sincrónico acá adentro: mientras duraba,
        # el bot no atendía a NADIE. `_on_text` ya usaba `_run_blocking`; el
        # monitor se había quedado afuera, y es el que peor lo llevaba —
        # corre cada 60 s hable alguien o no, así que un cluster lento
        # congelaba el bot de forma periódica sin que nadie lo pidiera.
        #
        # Combinado con un `_run` que podía esperar para siempre (arreglado
        # en `ssh_gateway`), esto no era lentitud: era un cuelgue definitivo
        # del bot entero a partir de un solo trabajo.
        vencidos = await self._en_hilo(self._service.sweep_expired_pendings)
        # Pedidos que se quedaron esperando una respuesta que no llegó. Va
        # ANTES de los trabajos y fuera del guard del monitor: cerrar una
        # consulta abierta no depende de que haya seguimiento configurado.
        for chat_id, text in vencidos:
            try:
                await self._enviar(
                    self._bot_sender(context, chat_id), text, chat_id=chat_id
                )
            except Exception as exc:
                logger.error("No pude avisar el vencimiento a chat_id=%s: %s", chat_id, exc)

        if self._job_monitor is None:  # pragma: no cover - guard defensivo
            return
        for note in await self._en_hilo(self._job_monitor.poll_and_notify):
            try:
                await self._enviar(
                    self._bot_sender(context, note.chat_id),
                    note.text,
                    chat_id=note.chat_id,
                    monospace=note.monospace,
                )
            except Exception as exc:
                # Sin acuse: el trabajo sigue activo y el próximo tick lo
                # reintenta. Antes el monitor ya lo había dado por avisado
                # antes de llegar acá, así que este `except` loguea… y el
                # aviso se perdía para siempre.
                logger.error("No pude notificar a chat_id=%s: %s", note.chat_id, exc)
                continue
            if note.acuse is not None:
                await self._en_hilo(self._job_monitor.confirm_delivery, note.acuse)

    @staticmethod
    def _bot_sender(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
        """Adapta `context.bot.send_message` a la firma que espera `_enviar`.

        El monitor no tiene un `Chat` a mano (no está contestando un update,
        está avisando por su cuenta), así que manda por id.
        """
        async def enviar(cuerpo: str, **kwargs):
            return await context.bot.send_message(
                chat_id=chat_id, text=cuerpo, **kwargs
            )

        return enviar
