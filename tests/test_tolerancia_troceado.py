"""El límite de 4096 de Telegram se respeta en el borde (T9).

Parte de la batería de fallos inyectados de `docs/plan_tolerancia_fallos.md`.
El resto de la suite prueba que el sistema FUNCIONA; esta prueba que AGUANTA,
que es otra cosa: acá todos los dobles fallan a propósito. Ninguno de estos
tests necesita cluster, ni Ollama, ni red — y esa es la razón de que puedan
existir, y también de que no existieran: las fallas que cubren no se
reproducen pidiéndole cosas a un sistema sano.
"""
import asyncio
from becario.application.context import Reply
from becario.presentation.telegram_bot import TelegramBot, _LIMITE_TELEGRAM, _trocear
from .test_telegram_bot import FAKE_TOKEN, FakeChat, FakeChatLog, FakeService



# ---------------------------------------------------------------------------
# T9 — el límite de Telegram se respeta en el borde
# ---------------------------------------------------------------------------


class TestTroceado:
    def test_texto_corto_no_se_toca(self):
        assert _trocear("hola") == ["hola"]

    def test_corta_por_lineas_y_ninguna_pieza_excede(self):
        texto = "\n".join(f"linea {i:04d} " + "x" * 60 for i in range(200))
        trozos = _trocear(texto)

        assert len(trozos) > 1
        assert all(len(t) <= _LIMITE_TELEGRAM for t in trozos)
        # Nada se pierde ni se duplica: partir una tabla no puede comerse
        # una fila.
        assert "\n".join(trozos) == texto

    def test_una_linea_gigante_se_parte_igual(self):
        texto = "y" * 9000
        trozos = _trocear(texto)
        assert len(trozos) == 3
        assert "".join(trozos) == texto

    def test_el_escape_html_cuenta_para_el_limite(self):
        """El caso que un `len()` crudo deja pasar: `&` escapa a `&amp;`,
        así que 3000 caracteres crudos son 15000 enviados."""
        def medir(t):
            import html

            return len(html.escape(t)) + len("<pre></pre>")

        texto = "&" * 3000
        trozos = _trocear(texto, medir)

        assert len(trozos) > 1
        for t in trozos:
            assert len(f"<pre>{__import__('html').escape(t)}</pre>") <= _LIMITE_TELEGRAM


class TestEnvioTroceado:
    def test_una_reply_larga_llega_en_varios_mensajes(self):
        chat_log = FakeChatLog()
        texto = "\n".join(f"paso {i}: " + "z" * 80 for i in range(150))
        bot = TelegramBot(
            token=FAKE_TOKEN, service=FakeService(Reply(text=texto)), chat_log=chat_log
        )
        chat = FakeChat(chat_id=555)

        asyncio.run(
            bot._enviar(chat.send_message, texto, chat_id=555)
        )

        assert len(chat.sent) > 1, "no troceó: Telegram habría rechazado el envío"
        assert all(len(m) <= _LIMITE_TELEGRAM for m in chat.sent)
        # La bitácora guarda el contenido, no en cuántos pedazos viajó.
        assert chat_log.entries == [(555, "bot", texto)]

    def test_los_botones_van_en_el_ultimo_trozo(self):
        enviados = []

        async def enviar(cuerpo, **kwargs):
            enviados.append((cuerpo, kwargs.get("reply_markup")))

        bot = TelegramBot(token=FAKE_TOKEN, service=FakeService(Reply(text="x")))
        texto = "\n".join("w" * 100 for _ in range(80))

        asyncio.run(bot._enviar(enviar, texto, chat_id=1, markup="BOTONES"))

        assert len(enviados) > 1
        assert [m for _, m in enviados[:-1]] == [None] * (len(enviados) - 1)
        assert enviados[-1][1] == "BOTONES"

    def test_monospace_envuelve_cada_trozo(self):
        chat = FakeChat(chat_id=555)
        bot = TelegramBot(token=FAKE_TOKEN, service=FakeService(Reply(text="x")))
        texto = "\n".join(f"{i:>6}  {i * 1.5:>14.6f}" for i in range(400))

        asyncio.run(
            bot._enviar(chat.send_message, texto, chat_id=555, monospace=True)
        )

        assert len(chat.sent) > 1
        # Cada mensaje tiene que ser HTML válido por sí solo: un <pre> que
        # abre en un mensaje y cierra en el siguiente no se renderiza.
        for m in chat.sent:
            assert m.startswith("<pre>") and m.endswith("</pre>")
            assert len(m) <= _LIMITE_TELEGRAM
