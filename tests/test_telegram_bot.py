"""Tests de la capa de presentación: bitácora de conversación.

Se ejercitan los handlers del bot con dobles de prueba (sin red ni
Telegram real): la bitácora tiene que registrar lo que entra y lo que
sale, y una falla de persistencia NUNCA puede cortar la conversación.
"""
import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

from telegram.error import BadRequest

from becario.application.job_monitor import JobAck, Notification
from becario.application.services import Reply
from becario.application.services import FOREIGN_CONFIRMATION_TEXT
from becario.presentation.telegram_bot import TelegramBot

# Token con formato válido para PTB; nunca se usa contra la red.
FAKE_TOKEN = "123456:TEST-token"


class FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id
        self.sent: list[str] = []
        self.actions: list[str] = []
        # Se arma para UN solo envío fallido: el aviso que viene después
        # tiene que poder salir, que es justo lo que se quiere probar.
        self.send_error: Exception | None = None

    async def send_message(self, text, reply_markup=None, parse_mode=None):
        self.sent.append(text)
        if self.send_error is not None:
            self.send_error, fallo = None, self.send_error
            raise fallo

    async def send_chat_action(self, action):
        self.actions.append(str(action))


class FakeUpdate:
    def __init__(self, chat: FakeChat, user_id: int, text: str) -> None:
        self.effective_chat = chat
        self.effective_user = SimpleNamespace(id=user_id)
        self.message = SimpleNamespace(text=text)


class FakeService:
    """Devuelve siempre la misma Reply y registra las llamadas."""

    def __init__(self, reply: Reply) -> None:
        self._reply = reply
        self.calls: list[tuple] = []

    def handle_text(self, chat_id: int, user_id: int, text: str) -> Reply:
        self.calls.append((chat_id, user_id, text))
        return self._reply

    def confirm(self, token: str, requester_id: int) -> Reply:
        self.calls.append(("confirm", token, requester_id))
        return self._reply

    def reject(self, token: str, requester_id: int) -> Reply:
        self.calls.append(("reject", token, requester_id))
        return self._reply

    def start_modification(self, token: str, requester_id: int, chat_id: int) -> Reply:
        # `chat_id` se registra a propósito: mientras el doble lo descartaba,
        # la presentación pasó durante meses el objeto `Chat` entero en vez
        # del entero y ningún test lo vio (el aviso de vencimiento moría
        # después, en el `send_message` del barrido).
        self.calls.append(("modify", token, requester_id, chat_id))
        return self._reply

    def sweep_expired_pendings(self) -> list[tuple[int, str]]:
        """Pedidos vencidos a avisar. Los tests que no lo ejercitan devuelven
        vacío; `TestExpiryNotice` lo sobreescribe."""
        return list(getattr(self, "expired", []))


class FakeVoiceFile:
    async def download_to_drive(self, path):
        Path(path).write_bytes(b"OGG-fake")


class FakeVoice:
    async def get_file(self):
        return FakeVoiceFile()


class FakeVoiceUpdate:
    def __init__(self, chat: FakeChat, user_id: int) -> None:
        self.effective_chat = chat
        self.effective_user = SimpleNamespace(id=user_id)
        self.message = SimpleNamespace(voice=FakeVoice())


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio_bytes: bytes) -> str:
        return self._text


class FakeCallbackQuery:
    def __init__(self, chat: FakeChat, user_id: int, data: str) -> None:
        self.from_user = SimpleNamespace(id=user_id)
        self.data = data
        self.message = SimpleNamespace(chat=chat)
        self.answers: list[tuple] = []
        # Telegram caduca los callbacks; `answer()` sobre uno vencido
        # levanta. El doble lo sabe hacer porque pasó en producción.
        self.answer_error: Exception | None = None

    async def answer(self, text=None, show_alert=False):
        if self.answer_error is not None:
            raise self.answer_error
        self.answers.append((text, show_alert))

    async def edit_message_reply_markup(self, reply_markup=None):
        pass


class FakeBotAPI:
    """Doble de `context.bot` para el tick del monitor."""

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self.sent: list[tuple] = []

    async def send_message(self, chat_id, text, parse_mode=None):
        if self._fail:
            raise RuntimeError("red caída")
        self.sent.append((chat_id, text))


class FakeChatLog:
    def __init__(self) -> None:
        self.entries: list[tuple[int, str, str]] = []

    def add(self, chat_id: int, role: str, text: str) -> None:
        self.entries.append((chat_id, role, text))

    def recent(self, chat_id: int, limit: int = 50) -> list[dict]:
        rows = [
            {"chat_id": c, "role": r, "text": t}
            for (c, r, t) in self.entries
            if c == chat_id
        ]
        return rows[-limit:]


class FailingChatLog(FakeChatLog):
    """Simula una base rota: cada intento de escritura falla."""

    def add(self, chat_id: int, role: str, text: str) -> None:
        raise sqlite3.OperationalError("database is locked")


def _make_bot(reply: Reply, chat_log, transcriber=None) -> tuple[TelegramBot, FakeService]:
    service = FakeService(reply)
    bot = TelegramBot(
        token=FAKE_TOKEN, service=service, chat_log=chat_log, transcriber=transcriber
    )
    return bot, service


class TestChatLogging:
    def test_logs_user_message_and_bot_reply(self):
        chat_log = FakeChatLog()
        bot, _ = _make_bot(Reply(text="listo, enviado"), chat_log)
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="mandá el trabajo"), None))

        assert chat_log.entries == [
            (555, "user", "mandá el trabajo"),
            (555, "bot", "listo, enviado"),
        ]
        assert chat.sent == ["listo, enviado"]

    def test_monospace_reply_logs_raw_text(self):
        # La bitácora guarda el contenido, no el envoltorio <pre> del canal.
        chat_log = FakeChatLog()
        bot, _ = _make_bot(Reply(text="col1  col2", monospace=True), chat_log)
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="estado"), None))

        assert (555, "bot", "col1  col2") in chat_log.entries
        assert chat.sent == ["<pre>col1  col2</pre>"]

    def test_chat_log_failure_does_not_break_conversation(self, caplog):
        bot, service = _make_bot(Reply(text="respuesta"), FailingChatLog())
        chat = FakeChat(chat_id=555)

        with caplog.at_level(logging.WARNING):
            asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="hola"), None))

        # La conversación siguió: el servicio atendió y la respuesta salió.
        assert service.calls == [(555, 7, "hola")]
        assert chat.sent == ["respuesta"]
        assert any("bitácora" in rec.message for rec in caplog.records)

    def test_without_chat_log_nothing_breaks(self):
        bot, service = _make_bot(Reply(text="respuesta"), chat_log=None)
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="hola"), None))

        assert chat.sent == ["respuesta"]

    def test_recent_reads_back_in_order(self):
        chat_log = FakeChatLog()
        bot, _ = _make_bot(Reply(text="ok"), chat_log)
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="primero"), None))
        asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="segundo"), None))

        roles = [(r["role"], r["text"]) for r in chat_log.recent(555)]
        assert roles == [
            ("user", "primero"),
            ("bot", "ok"),
            ("user", "segundo"),
            ("bot", "ok"),
        ]


class TestVoiceChatLogging:
    def test_transcription_logged_as_user_and_replies_as_bot(self):
        chat_log = FakeChatLog()
        bot, service = _make_bot(
            Reply(text="ok"), chat_log,
            transcriber=FakeTranscriber("  relanzá el trabajo  "),
        )
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_voice(FakeVoiceUpdate(chat, user_id=7), None))

        # La transcripción (sin espacios sobrantes) queda como mensaje del
        # usuario; el eco «Entendí» y la respuesta final, como bot.
        assert chat_log.entries == [
            (555, "user", "relanzá el trabajo"),
            (555, "bot", "🎙️ Entendí: «relanzá el trabajo»"),
            (555, "bot", "ok"),
        ]
        assert service.calls  # el servicio efectivamente atendió el pedido

    def test_transcriber_not_configured_notice_logged_as_bot(self):
        chat_log = FakeChatLog()
        bot, _ = _make_bot(Reply(text="ok"), chat_log, transcriber=None)
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_voice(FakeVoiceUpdate(chat, user_id=7), None))

        assert chat_log.entries == [
            (555, "bot", "🎙️ La transcripción de audio no está configurada."),
        ]

    def test_empty_transcription_notice_logged_as_bot(self):
        chat_log = FakeChatLog()
        bot, service = _make_bot(
            Reply(text="ok"), chat_log, transcriber=FakeTranscriber("   ")
        )
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_voice(FakeVoiceUpdate(chat, user_id=7), None))

        assert chat_log.entries == [
            (555, "bot", "🎙️ No pude entender el audio, probá de nuevo."),
        ]
        assert service.calls == []  # nunca llegó al servicio


class TestCallbackChatLogging:
    def _run_callback(self, data: str, reply_text: str):
        chat_log = FakeChatLog()
        bot, service = _make_bot(Reply(text=reply_text), chat_log)
        chat = FakeChat(chat_id=555)
        query = FakeCallbackQuery(chat, user_id=7, data=data)
        update = SimpleNamespace(callback_query=query)
        asyncio.run(bot._on_callback(update, None))
        return chat_log, chat, service

    def test_confirm_logs_user_decision_then_bot_reply(self):
        chat_log, chat, service = self._run_callback("confirm:tok1", "🚀 Enviado.")

        assert chat_log.entries == [
            (555, "user", "confirmar"),
            (555, "bot", "🚀 Enviado."),
        ]
        assert chat.sent == ["🚀 Enviado."]
        assert ("confirm", "tok1", 7) in service.calls

    def test_cancel_logs_user_decision_then_bot_reply(self):
        chat_log, chat, service = self._run_callback("cancel:tok1", "Cancelado.")

        assert chat_log.entries == [
            (555, "user", "cancelar"),
            (555, "bot", "Cancelado."),
        ]
        assert ("reject", "tok1", 7) in service.calls

    def test_modify_logs_user_decision(self):
        chat_log, _, service = self._run_callback("modify:tok1", "¿Qué cambiamos?")

        assert chat_log.entries[0] == (555, "user", "modificar")
        assert ("modify", "tok1", 7, 555) in service.calls


class TestAnExpiredCallbackIsNotACrash:
    """Telegram caduca los callbacks: `answer()` sobre uno viejo levanta
    `BadRequest("Query is too old…")`.

    Pasó dos veces en una sola sesión, sobre una tarjeta de quince minutos.
    El usuario recibió «se me rompió algo y no ejecuté nada» con un código
    de incidente —que suena a bot roto— cuando lo único que había pasado es
    que se tardó: la confirmación ya había vencido por su propio TTL y
    `reject` tenía listo un «⌛ Esta confirmación expiró», que explica qué
    pasó y nunca llegó a mandarse.
    """

    def _vencido(self, reply_text: str, data: str = "cancel:tok1"):
        chat_log = FakeChatLog()
        bot, service = _make_bot(Reply(text=reply_text), chat_log)
        chat = FakeChat(chat_id=555)
        query = FakeCallbackQuery(chat, user_id=7, data=data)
        query.answer_error = BadRequest("Query is too old and response timeout expired")
        asyncio.run(bot._on_callback(SimpleNamespace(callback_query=query), None))
        return chat, service

    def test_the_useful_message_still_arrives(self):
        chat, service = self._vencido("⌛ Esta confirmación expiró o ya fue usada.")

        # Lo que importa: el acuse fallido no se llevó puesta la respuesta.
        assert chat.sent == ["⌛ Esta confirmación expiró o ya fue usada."]
        assert ("reject", "tok1", 7) in service.calls

    def test_a_stale_ack_never_becomes_an_incident(self):
        chat, _ = self._vencido("⌛ Esta confirmación expiró o ya fue usada.")

        assert not any("incidente" in t.lower() for t in chat.sent)

    def test_a_foreign_touch_that_cannot_be_answered_stays_quiet(self):
        # El aviso de toque ajeno va SOLO al alert de quien tocó. Si el
        # callback venció no hay dónde ponerlo: no se ensucia el chat de
        # quien sí puede decidir, y tampoco se cae nada.
        chat, _ = self._vencido(FOREIGN_CONFIRMATION_TEXT, data="confirm:tok1")

        assert chat.sent == []


class TestADeliveryFailureAfterConfirmingNeverSaysNothingRan:
    """El riesgo latente de la misma línea, y el peor de los seis.

    `confirm` ejecuta la acción real —el `sbatch` sale por SSH— y recién
    después la presentación entrega el resultado. Si esa entrega revienta,
    el handler global contesta «no ejecuté nada, podés volver a intentarlo».
    Sobre un trabajo ya encolado, esa frase son dos trabajos en la cola y
    dos veces las horas de cómputo por un error que el usuario nunca vio.
    """

    def _entrega_rota(self, data: str):
        chat_log = FakeChatLog()
        bot, service = _make_bot(Reply(text="🚀 Enviado. Job 4321."), chat_log)
        chat = FakeChat(chat_id=555)
        chat.send_error = RuntimeError("red caída")
        query = FakeCallbackQuery(chat, user_id=7, data=data)
        try:
            asyncio.run(bot._on_callback(SimpleNamespace(callback_query=query), None))
        except RuntimeError as exc:
            return chat, service, exc
        return chat, service, None

    def test_it_says_it_cannot_be_sure_instead_of_saying_nothing_ran(self):
        chat, service, exc = self._entrega_rota("confirm:tok1")

        assert exc is None, "no puede subir al handler global, que diría otra cosa"
        assert ("confirm", "tok1", 7) in service.calls
        # El primer envío es el que falló; el segundo es el aviso.
        aviso = chat.sent[-1]
        assert "no puedo asegurarte si llegó a hacerse" in aviso
        assert "No la repitas a ciegas" in aviso
        assert "incidente" in aviso.lower()

    def test_a_cancel_that_cannot_be_delivered_goes_to_the_global_handler(self):
        # ✖️ no toca el cluster: ahí «no ejecuté nada» es cierto, y el
        # handler global es quien corresponde. Fallar hacia el lado
        # ambiguo cuando no hay ambigüedad también es mentir.
        _, _, exc = self._entrega_rota("cancel:tok1")

        assert isinstance(exc, RuntimeError)


class TestCallbackPasaTiposDeclarados:
    """La frontera presentación↔aplicación, con los tipos que declara.

    Los tests del servicio pasan `chat_id=1` a mano, así que verifican la
    función pero nunca la LLAMADA. Este es el hueco por el que se coló que
    `_on_callback` mandara un `telegram.Chat` donde la firma pide un `int`:
    943 tests en verde y el aviso de vencimiento muerto en producción,
    porque el `send_message` del barrido fallaba y el `except` lo tapaba.
    """

    def _modify(self, con_mensaje: bool = True):
        bot, service = _make_bot(Reply(text="¿Qué cambiamos?"), FakeChatLog())
        query = FakeCallbackQuery(FakeChat(chat_id=555), user_id=7, data="modify:tok1")
        if not con_mensaje:
            query.message = None
        asyncio.run(bot._on_callback(SimpleNamespace(callback_query=query), None))
        return [c for c in service.calls if c[0] == "modify"][0]

    def test_modify_manda_el_id_del_chat_no_el_objeto(self):
        _, _, _, chat_id = self._modify()

        assert chat_id == 555
        # `bool` es subclase de `int` y un `Chat` no lo es: lo que importa
        # es que sea el entero que `send_message` sabe usar.
        assert type(chat_id) is int

    def test_sin_mensaje_manda_el_centinela_de_no_avisar(self):
        # `query.message` es None en un mensaje viejo: no hay a quién
        # avisarle. 0 es el centinela que `sweep_expired_pendings` ya trata
        # como "vencer callado" (`if edit.chat_id:`).
        _, _, _, chat_id = self._modify(con_mensaje=False)

        assert chat_id == 0


class TestMonitorTickChatLogging:
    """Se usa el `Notification` REAL y no un `SimpleNamespace`: un doble con
    forma propia deja de avisar cuando el tipo verdadero crece (fue el caso
    de `acuse`, agregado para que un envío fallido no se lleve el aviso)."""

    def _make_monitored_bot(self, note) -> tuple[TelegramBot, list]:
        bot, _ = _make_bot(Reply(text="ok"), FakeChatLog())
        acusados: list = []
        # Se inyecta el monitor a mano para no requerir el extra job-queue.
        bot._job_monitor = SimpleNamespace(
            poll_and_notify=lambda: [note],
            confirm_delivery=acusados.append,
        )
        return bot, acusados

    @staticmethod
    def _note(**kwargs) -> Notification:
        base = dict(
            chat_id=999,
            text="✅ Terminó el trabajo 42.",
            acuse=JobAck(job_id="42", owner_id=7, job_name="zr", estado="completado"),
        )
        return Notification(**{**base, **kwargs})

    def test_delivered_notification_is_logged(self):
        bot, acusados = self._make_monitored_bot(self._note())
        context = SimpleNamespace(bot=FakeBotAPI())

        asyncio.run(bot._on_monitor_tick(context))

        assert bot._chat_log.entries == [(999, "bot", "✅ Terminó el trabajo 42.")]

    def test_delivered_notification_is_acked(self):
        # El asiento va DESPUÉS del envío: es lo que convierte "avisé" en
        # "el aviso salió".
        bot, acusados = self._make_monitored_bot(self._note())
        context = SimpleNamespace(bot=FakeBotAPI())

        asyncio.run(bot._on_monitor_tick(context))

        assert [a.job_id for a in acusados] == ["42"]

    def test_failed_send_is_not_logged(self, caplog):
        # Contrato de orden: la bitácora solo registra lo efectivamente
        # entregado. Si el envío falla, no debe quedar rastro del mensaje.
        bot, acusados = self._make_monitored_bot(self._note())
        context = SimpleNamespace(bot=FakeBotAPI(fail=True))

        with caplog.at_level(logging.ERROR):
            asyncio.run(bot._on_monitor_tick(context))

        assert bot._chat_log.entries == []
        assert any("No pude notificar" in rec.message for rec in caplog.records)

    def test_failed_send_is_not_acked(self, caplog):
        """El corazón de T4: sin acuse, el trabajo sigue activo y el
        próximo tick lo reintenta. Antes el monitor ya lo había dado por
        avisado, así que el aviso se perdía para siempre."""
        bot, acusados = self._make_monitored_bot(self._note())
        context = SimpleNamespace(bot=FakeBotAPI(fail=True))

        with caplog.at_level(logging.ERROR):
            asyncio.run(bot._on_monitor_tick(context))

        assert acusados == []

    def test_harvest_notification_is_not_acked(self):
        # La cosecha del barrido es un SEGUNDO mensaje del mismo trabajo:
        # acusarla otra vez duplicaría la fila del historial.
        bot, acusados = self._make_monitored_bot(self._note(acuse=None))
        context = SimpleNamespace(bot=FakeBotAPI())

        asyncio.run(bot._on_monitor_tick(context))

        assert acusados == []
        assert bot._chat_log.entries  # pero el mensaje sí salió


class TestTypingIndicator:
    """El trabajo bloqueante corre en un hilo con el indicador
    «escribiendo…» activo (ver `_run_blocking`)."""

    def test_typing_action_sent_while_processing_text(self):
        bot, service = _make_bot(Reply(text="respuesta"), FakeChatLog())
        chat = FakeChat(chat_id=555)

        asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="hola"), None))

        assert len(chat.actions) >= 1
        assert "typing" in chat.actions[0]
        assert chat.sent == ["respuesta"]  # la respuesta llega igual

    def test_slow_service_keeps_loop_responsive(self):
        # Un servicio lento NO congela el event loop: mientras el hilo
        # trabaja, el loop puede ejecutar otras corutinas.
        class SlowService(FakeService):
            def handle_text(self, chat_id, user_id, text):
                time.sleep(0.3)
                return super().handle_text(chat_id, user_id, text)

        service = SlowService(Reply(text="tardé pero llegué"))
        bot = TelegramBot(token=FAKE_TOKEN, service=service, chat_log=FakeChatLog())
        chat = FakeChat(chat_id=555)
        loop_alive = []

        async def _both():
            async def _heartbeat():
                for _ in range(3):
                    await asyncio.sleep(0.05)
                    loop_alive.append(True)
            await asyncio.gather(
                bot._on_text(FakeUpdate(chat, user_id=7, text="hola"), None),
                _heartbeat(),
            )

        asyncio.run(_both())
        # Con el handler síncrono de antes, el heartbeat no habría corrido
        # hasta después de los 0.3s del servicio.
        assert loop_alive == [True, True, True]
        assert chat.sent == ["tardé pero llegué"]

    def test_typing_failure_does_not_break_reply(self):
        class NoActionChat(FakeChat):
            async def send_chat_action(self, action):
                raise RuntimeError("chat action no soportado")

        bot, _ = _make_bot(Reply(text="ok"), FakeChatLog())
        chat = NoActionChat(chat_id=555)

        asyncio.run(bot._on_text(FakeUpdate(chat, user_id=7, text="hola"), None))

        assert chat.sent == ["ok"]
