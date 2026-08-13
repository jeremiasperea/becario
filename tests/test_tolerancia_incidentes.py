"""Red de contención: ninguna excepción termina en silencio (T1).

Parte de la batería de fallos inyectados de `docs/plan_tolerancia_fallos.md`.
El resto de la suite prueba que el sistema FUNCIONA; esta prueba que AGUANTA,
que es otra cosa: acá todos los dobles fallan a propósito. Ninguno de estos
tests necesita cluster, ni Ollama, ni red — y esa es la razón de que puedan
existir, y también de que no existieran: las fallas que cubren no se
reproducen pidiéndole cosas a un sistema sano.
"""
import asyncio
import logging
import pytest
from becario.application.context import Reply
from becario.application.services import BecarioService
from becario.domain.models import Intent
from becario.incidentes import nuevo_incidente
from becario.infrastructure.storage import InMemoryConfirmationStore
from becario.presentation.telegram_bot import TelegramBot
from .test_service import (
    ALICE,
    FakeCalcInputGenerator,
    FakeCalcRuns,
    FakeClusterGatewayFactory,
    FakeHistory,
    FakeJobTracker,
    FakeRouter,
    FakeStructureBuilder,
    FakeUserRegistry,
    RoutedRequest,
)
from .test_telegram_bot import FAKE_TOKEN, FakeChat, FakeChatLog, FakeService



# ---------------------------------------------------------------------------
# T1 — ninguna excepción termina en silencio
# ---------------------------------------------------------------------------


class ChatQueRompe(FakeChat):
    """Telegram rechaza el envío: es lo que se cree que pasó con los
    mensajes 23 y 88 de la bitácora."""

    async def send_message(self, text, reply_markup=None, parse_mode=None):
        raise RuntimeError("Bad Request: message is too long")


class TestRedDeContencion:
    def test_el_handler_de_errores_esta_registrado(self):
        # El test más aburrido del archivo y el que más importa: sin el
        # registro, todo lo demás de esta clase prueba código muerto.
        bot = TelegramBot(token=FAKE_TOKEN, service=FakeService(Reply(text="x")))
        assert bot._app.error_handlers, "no hay error handler registrado"

    def test_una_excepcion_le_avisa_al_usuario_con_un_incidente(self, caplog):
        chat_log = FakeChatLog()
        bot = TelegramBot(
            token=FAKE_TOKEN, service=FakeService(Reply(text="x")), chat_log=chat_log
        )
        chat = FakeChat(chat_id=555)
        update = type("U", (), {"effective_chat": chat})()
        context = type("C", (), {"error": ValueError("algo se rompió")})()

        with caplog.at_level(logging.ERROR):
            asyncio.run(bot._on_error(update, context))

        # 1) El usuario recibió ALGO. Antes recibía silencio.
        assert len(chat.sent) == 1
        aviso = chat.sent[0]
        assert "no ejecuté nada" in aviso

        # 2) El id del incidente está en el chat Y en el log: ese es el
        #    puente que no existía entre "no me contestó" y el traceback.
        incidente = aviso.split("incidente ")[1].strip()
        assert len(incidente) == 8
        assert any(incidente in rec.message for rec in caplog.records)

        # 3) Y quedó en la bitácora, como cualquier otra respuesta.
        assert (555, "bot", aviso) in chat_log.entries

    def test_si_tampoco_se_puede_avisar_no_hay_bucle(self, caplog):
        # Una excepción DENTRO del error handler vuelve al error handler.
        # Este test existe para fijar que ese camino está cortado.
        bot = TelegramBot(token=FAKE_TOKEN, service=FakeService(Reply(text="x")))
        update = type("U", (), {"effective_chat": ChatQueRompe(chat_id=555)})()
        context = type("C", (), {"error": ValueError("x")})()

        with caplog.at_level(logging.ERROR):
            asyncio.run(bot._on_error(update, context))  # no debe propagar

        assert any("tampoco pude avisarle" in rec.message for rec in caplog.records)

    def test_un_error_sin_chat_no_revienta(self):
        # Errores del propio job_queue: no hay a quién avisarle, queda el log.
        bot = TelegramBot(token=FAKE_TOKEN, service=FakeService(Reply(text="x")))
        update = type("U", (), {"effective_chat": None})()
        context = type("C", (), {"error": ValueError("x")})()

        asyncio.run(bot._on_error(update, context))  # no debe propagar


# ---------------------------------------------------------------------------
# T1 (segunda mitad) — una excepción no evapora un plan confirmado
# ---------------------------------------------------------------------------


@pytest.fixture()
def servicio():
    factory = FakeClusterGatewayFactory()
    service = BecarioService(
        router=FakeRouter(),
        registry=FakeUserRegistry([ALICE]),
        cluster_factory=factory,
        history=FakeHistory(),
        confirmations=InMemoryConfirmationStore(ttl_seconds=600),
        structures=FakeStructureBuilder(),
        job_tracker=FakeJobTracker(),
        calc_inputs=FakeCalcInputGenerator(),
        potcar_dir="/potcars",
        remote_base="becario_runs",
        calc_runs=FakeCalcRuns(),
    )
    return service, factory


def _token_de_envio(service) -> str:
    service._router.next = RoutedRequest(
        intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
    )
    return service.handle_text(
        chat_id=1, user_id=ALICE.telegram_user_id, text="corré esto"
    ).confirmation_token


class TestFalloTrasConfirmar:
    def test_el_usuario_se_entera_y_lo_mandan_a_verificar(self, servicio, caplog):
        service, factory = servicio
        token = _token_de_envio(service)

        def revienta(ctx, action):
            raise RuntimeError("SSH murió a mitad del sbatch")

        service._step_executors = lambda: {Intent.SUBMIT_SLURM: revienta}

        with caplog.at_level(logging.ERROR):
            reply = service.confirm(token, requester_id=ALICE.telegram_user_id)

        assert reply.ok is False
        # Lo que NO tiene que decir: "volvé a intentar". El sbatch pudo
        # haber salido antes de la excepción.
        assert "No la repitas a ciegas" in reply.text
        assert "estado de mis trabajos" in reply.text
        incidente = reply.text.split("incidente: ")[1].strip()
        assert any(incidente in rec.message for rec in caplog.records)

    def test_el_token_NO_se_repone(self, servicio):
        """La parte contraintuitiva, y la razón de que el arreglo no sea
        `peek` → ejecutar → `pop`.

        Reponer el token después de un fallo parece lo amable, pero `pop`
        es atómico justamente para que dos toques de ✅ no manden dos
        `sbatch`. Si la excepción saltó DESPUÉS del envío, reponerlo le
        pone al usuario un botón que duplica un trabajo ya encolado.
        """
        service, _ = servicio
        token = _token_de_envio(service)
        service._step_executors = lambda: {
            Intent.SUBMIT_SLURM: lambda ctx, action: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
        }

        service.confirm(token, requester_id=ALICE.telegram_user_id)
        segundo = service.confirm(token, requester_id=ALICE.telegram_user_id)

        assert "ya se usó" in segundo.text  # consumido, no repuesto

    def test_sin_excepcion_nada_cambia(self, servicio):
        # La red de contención no puede alterar el camino feliz.
        service, factory = servicio
        token = _token_de_envio(service)
        reply = service.confirm(token, requester_id=ALICE.telegram_user_id)
        assert "4242" in reply.text
        assert len(factory.gateways["alice"].submitted) == 1


# ---------------------------------------------------------------------------
# Incidentes
# ---------------------------------------------------------------------------


class TestIncidentes:
    def test_son_unicos_y_cortos(self):
        ids = {nuevo_incidente() for _ in range(500)}
        assert len(ids) == 500
        assert all(len(i) == 8 for i in ids)
