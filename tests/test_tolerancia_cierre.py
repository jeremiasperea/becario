"""El bot cierra sus conexiones al terminar (T8).

Parte de la batería de fallos inyectados de `docs/plan_tolerancia_fallos.md`.
El resto de la suite prueba que el sistema FUNCIONA; esta prueba que AGUANTA,
que es otra cosa: acá todos los dobles fallan a propósito. Ninguno de estos
tests necesita cluster, ni Ollama, ni red — y esa es la razón de que puedan
existir, y también de que no existieran: las fallas que cubren no se
reproducen pidiéndole cosas a un sistema sano.
"""
import logging
import pytest
from becario.application.context import Reply
from becario.domain.models import CommandFailureReason, CommandResult, Intent
from becario.infrastructure.ssh_gateway import SSHClusterGateway
from becario.presentation.telegram_bot import TelegramBot
from .test_service import ALICE, RoutedRequest, env
from .test_telegram_bot import FAKE_TOKEN, FakeService


# ---------------------------------------------------------------------------
# T8 — cierre ordenado
# ---------------------------------------------------------------------------


class TestCierreOrdenado:
    def test_al_terminar_se_cierran_las_conexiones(self):
        """`close_all()` existía desde el principio y no lo llamaba nadie:
        cada `systemctl restart` dejaba los transportes SSH abiertos."""
        cerrados = []
        bot = TelegramBot(
            token=FAKE_TOKEN,
            service=FakeService(Reply(text="x")),
            al_cerrar=lambda: cerrados.append(1),
        )
        bot._app.run_polling = lambda **kw: None

        bot.run()

        assert cerrados == [1]

    def test_se_cierra_igual_si_el_polling_revienta(self):
        # Un `systemctl stop` sale por acá; un crash también, y las
        # conexiones hay que soltarlas en los dos casos.
        cerrados = []
        bot = TelegramBot(
            token=FAKE_TOKEN,
            service=FakeService(Reply(text="x")),
            al_cerrar=lambda: cerrados.append(1),
        )
        def revienta(**kw):
            raise KeyboardInterrupt
        bot._app.run_polling = revienta

        with pytest.raises(KeyboardInterrupt):
            bot.run()

        assert cerrados == [1]

    def test_un_cierre_que_falla_no_tapa_el_motivo_real(self, caplog):
        bot = TelegramBot(
            token=FAKE_TOKEN,
            service=FakeService(Reply(text="x")),
            al_cerrar=lambda: (_ for _ in ()).throw(RuntimeError("SSH ya muerto")),
        )
        bot._app.run_polling = lambda **kw: None

        with caplog.at_level(logging.ERROR):
            bot.run()  # no debe propagar

        assert any("cierre ordenado" in r.message for r in caplog.records)
