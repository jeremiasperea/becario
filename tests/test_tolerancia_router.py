"""Un fallo del modelo no se disfraza de «no te entendí» (T5).

Parte de la batería de fallos inyectados de `docs/plan_tolerancia_fallos.md`.
El resto de la suite prueba que el sistema FUNCIONA; esta prueba que AGUANTA,
que es otra cosa: acá todos los dobles fallan a propósito. Ninguno de estos
tests necesita cluster, ni Ollama, ni red — y esa es la razón de que puedan
existir, y también de que no existieran: las fallas que cubren no se
reproducen pidiéndole cosas a un sistema sano.
"""
import httpx
import pytest
from becario.application.services import BecarioService, HELP_TEXT
from becario.domain.models import Intent, RouterFailureReason, RouterUnavailableError
from becario.infrastructure.ollama_router import OllamaRouter
from becario.infrastructure.storage import InMemoryConfirmationStore
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
)



# ---------------------------------------------------------------------------
# T5 — un fallo del modelo no se disfraza de "no te entendí"
# ---------------------------------------------------------------------------


class RouterCaido:
    """Router que siempre falla por infraestructura, con la causa que se le
    pida. No es "el modelo no entendió": es que nunca llegó a leer nada."""

    def __init__(self, reason, timeout_seconds=None) -> None:
        self._exc = RouterUnavailableError(
            reason, "simulado", timeout_seconds=timeout_seconds
        )

    def _falla(self, *a, **k):
        raise self._exc

    route = decompose = extract_params = extract_edit = _falla


def _servicio_con_router(router):
    return BecarioService(
        router=router,
        registry=FakeUserRegistry([ALICE]),
        cluster_factory=FakeClusterGatewayFactory(),
        history=FakeHistory(),
        confirmations=InMemoryConfirmationStore(ttl_seconds=600),
        structures=FakeStructureBuilder(),
        job_tracker=FakeJobTracker(),
        calc_inputs=FakeCalcInputGenerator(),
        potcar_dir="/potcars",
        remote_base="becario_runs",
        calc_runs=FakeCalcRuns(),
    )


class TestMensajeHonestoDelRouter:
    def test_timeout_dice_tiempo_de_espera_y_no_culpa_al_usuario(self):
        service = _servicio_con_router(
            RouterCaido(RouterFailureReason.TIMEOUT, timeout_seconds=180)
        )

        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="relajá el ZrO2"
        )

        assert "Tiempo de espera agotado" in reply.text
        assert "180 segundos" in reply.text
        # Lo que importa que NO diga: el pedido estaba perfecto.
        assert reply.text != HELP_TEXT
        assert "No pude interpretar tu pedido" not in reply.text
        assert reply.ok is False

    def test_servidor_caido_manda_a_avisar_no_a_reescribir(self):
        service = _servicio_con_router(RouterCaido(RouterFailureReason.UNREACHABLE))

        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="listá mi home"
        )

        assert "no responde" in reply.text
        assert "avisale a quien administra" in reply.text
        # Un dato que calma: lo que ya está corriendo no se ve afectado.
        assert "siguen corriendo" in reply.text
        assert "No pude interpretar tu pedido" not in reply.text

    def test_los_tres_motivos_dan_mensajes_distintos(self):
        textos = {
            reason: _servicio_con_router(RouterCaido(reason))
            .handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
            .text
            for reason in RouterFailureReason
        }
        assert len(set(textos.values())) == 3, "dos motivos comparten mensaje"

    def test_un_fallo_del_modelo_no_ensucia_el_dataset_del_router(self):
        """No hubo decisión, así que no se registra ninguna: si se anotara
        como 'error', el dataset de fixtures (que alimenta los fixtures del
        tablero) se llenaría de fallos que no son del router.

        OJO con el doble: `_log_decision` llama `add` POSICIONAL. Un fake
        con `**kwargs` revienta, el `except` de `_log_decision` se lo come
        —es observabilidad, no puede tirar el mensaje— y el test pasa por el
        motivo equivocado. De ahí el control de abajo.
        """
        registrador = []

        class LogQueRegistra:
            def add(self, chat_id, user_id, text, steps_json, latency):
                registrador.append(text)
                return 1

            def set_outcome(self, decision_id, outcome):
                registrador.append(("outcome", outcome))

        # Control: con un router sano, este mismo doble SÍ registra. Sin
        # esto, la aserción de abajo no distingue "no se registró" de "el
        # doble está roto".
        sano = _servicio_con_router(FakeRouter())
        sano._decision_log = LogQueRegistra()
        sano.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="hola")
        assert registrador == ["hola"], "el doble no registra ni en el camino feliz"

        registrador.clear()
        caido = _servicio_con_router(RouterCaido(RouterFailureReason.TIMEOUT))
        caido._decision_log = LogQueRegistra()
        caido.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

        assert registrador == []


class TestMapeoDeExcepcionesHttpx:
    """El adaptador es quien sabe traducir httpx al motivo del dominio."""

    def _router(self):
        return OllamaRouter(base_url="http://localhost:11434", model="m", timeout=180)

    @pytest.mark.parametrize(
        "excepcion, esperado",
        [
            (httpx.ConnectTimeout("agotado"), RouterFailureReason.TIMEOUT),
            (httpx.ReadTimeout("agotado"), RouterFailureReason.TIMEOUT),
            (httpx.ConnectError("sin servidor"), RouterFailureReason.UNREACHABLE),
        ],
    )
    def test_traduce_la_causa(self, monkeypatch, excepcion, esperado):
        def revienta(*a, **k):
            raise excepcion

        monkeypatch.setattr(httpx, "post", revienta)

        with pytest.raises(RouterUnavailableError) as info:
            self._router().route("relajá el ZrO2")

        assert info.value.reason is esperado

    def test_el_timeout_viaja_en_la_excepcion(self, monkeypatch):
        # Para poder decirle al usuario CUÁNTO se esperó.
        monkeypatch.setattr(
            httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("x"))
        )
        with pytest.raises(RouterUnavailableError) as info:
            self._router().route("x")
        assert info.value.timeout_seconds == 180

    def test_el_backfill_es_la_excepcion_a_la_regla(self, monkeypatch):
        """`extract_structure` es una segunda pasada opcional: si el servidor
        se cae entre las dos llamadas, el plan de `route()` sigue sirviendo y
        no se puede tirar el pedido entero por la mejora."""
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("caído")),
        )
        assert self._router().extract_structure("slab de ZrO2") == {}


class TestElPlanSobreviveAlFallo:
    def test_un_fallo_editando_no_se_lleva_el_plan(self):
        """La promesa de `_apply_edit` es que solo un «cancelar» explícito
        descarta el plan. Vale también cuando quien falla es el bot."""
        router = RouterCaido(RouterFailureReason.TIMEOUT, timeout_seconds=180)
        service = _servicio_con_router(router)
        service._arm_pending_edit(
            ALICE.telegram_user_id, 1, [(Intent.SUBMIT_SLURM, {"nodos": 1})]
        )

        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="usá 2 nodos"
        )

        assert "Tiempo de espera agotado" in reply.text
        assert "Tu plan sigue esperando" in reply.text
        assert reply.awaiting_params is True
        # Y de verdad sigue en el estante, no solo en el mensaje.
        assert service._pending_edits.has(ALICE.telegram_user_id)
