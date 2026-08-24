"""Los tres «no puedo» del bot dejan rastro, y se distinguen entre sí.

El bot rechaza un pedido en tres momentos distintos —— antes de rutear, al
rutear, y con los params ya parseados —— y siguen siendo tres a propósito:
en el momento en que corre cada uno, la información de los otros dos no
existe todavía (ver `becario/application/rechazos.py`).

Lo que hasta acá NO se podía era saber CUÁL de los tres se cerró sin
reconocer la prosa de memoria: el pre-router dejaba su propio `logger.info`,
el del router quedaba enterrado en el `steps_json` del decision log, y el de
vocabulario no dejaba rastro ninguno. Estos tests fijan que los tres
registran el mismo evento con su origen marcado.

Y fijan lo otro, que importa igual: **los textos que ve el usuario no
cambiaron**. El rastro es para quien depura, no para quien pregunta.
"""
from __future__ import annotations

import logging

import pytest

from becario.application import rechazos
from becario.application.services import BecarioService, HELP_TEXT
from becario.domain.fuera_de_alcance import NO_SE_APILAR
from becario.domain.models import Intent, RoutedRequest
from becario.infrastructure.storage import InMemoryConfirmationStore
from tests.test_service import (
    ALICE,
    BOB,
    FakeCalcInputGenerator,
    FakeCalcRuns,
    FakeClusterGatewayFactory,
    FakeHistory,
    FakeJobTracker,
    FakeRouter,
    FakeStructureBuilder,
    FakeUserRegistry,
    _CONTCAR_ZR,
)

RUN_DIR = "/data/runs/zr_relax"


@pytest.fixture()
def env():
    router = FakeRouter()
    factory = FakeClusterGatewayFactory()
    service = BecarioService(
        router=router,
        registry=FakeUserRegistry([ALICE, BOB]),
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
    return service, router, factory


def _mandar(service, texto="x"):
    return service.handle_text(
        chat_id=1, user_id=ALICE.telegram_user_id, text=texto
    )


def _con_una_corrida(service, factory):
    """La corrida mínima que hace falta para llegar al portón de vocabulario:
    sin una corrida registrada el handler corta antes y nunca se pronuncia."""
    service._calc_runs.add(ALICE.telegram_user_id, "14", "Zr_relajacion", "{}", RUN_DIR)
    cluster = factory.for_identity(ALICE)
    cluster.remote_files[f"{RUN_DIR}/CONTCAR"] = _CONTCAR_ZR
    cluster.remote_files[f"{RUN_DIR}/OSZICAR"] = (
        "   1 F= -.17096101E+02 E0= -.17098019E+02  d E =-.17E+02\n"
    )
    return cluster


def _origenes(caplog):
    """Los `origen=` que quedaron en el log, en orden."""
    return [
        r.message.split("origen=")[1].split(" ")[0]
        for r in caplog.records
        if "rechazo origen=" in r.message
    ]


class TestCadaPortonDejaSuRastro:
    def test_el_pedido_imposible_se_marca_pre_router(self, env, caplog):
        # Corta ANTES de rutear: es lo único que lo hace valioso.
        service, *_ = env
        with caplog.at_level(logging.INFO):
            reply = _mandar(service, "armá Zr sobre W")
        assert _origenes(caplog) == [rechazos.PRE_ROUTER]
        assert reply.text == NO_SE_APILAR  # el texto NO cambió

    def test_el_router_sin_handler_se_marca_router(self, env, caplog):
        # El router se pronunció y no hay handler para lo que dijo.
        service, router, _ = env
        router.next = RoutedRequest(intent=Intent.UNKNOWN, params={})
        with caplog.at_level(logging.INFO):
            reply = _mandar(service)
        assert _origenes(caplog) == [rechazos.ROUTER]
        assert reply.text == HELP_TEXT  # el texto NO cambió

    def test_la_abstencion_de_vocabulario_se_marca_vocabulario(self, env, caplog):
        # El plan es válido; el dato pedido no está en el vocabulario.
        service, router, factory = env
        _con_una_corrida(service, factory)
        router.next = RoutedRequest(
            intent=Intent.QUERY_RESULTS, params={"formula": "Zr", "dato": "ninguno"}
        )
        with caplog.at_level(logging.INFO):
            reply = _mandar(service)
        assert _origenes(caplog) == [rechazos.VOCABULARIO]
        assert "no está entre ellos" in reply.text  # el texto NO cambió
        assert "a = " not in reply.text

    def test_el_detalle_del_vocabulario_dice_de_que_corrida_hablaba(self, env, caplog):
        # `origen=vocabulario` solo dice QUE se abstuvo. Para depurar hace
        # falta saber sobre qué corrida, que es lo que no se podía reconstruir.
        service, router, factory = env
        _con_una_corrida(service, factory)
        router.next = RoutedRequest(
            intent=Intent.QUERY_RESULTS, params={"formula": "Zr", "dato": "ninguno"}
        )
        with caplog.at_level(logging.INFO):
            _mandar(service)
        rechazo = next(r for r in caplog.records if "rechazo origen=" in r.message)
        assert RUN_DIR in rechazo.message


class TestLosTresSonUnaSolaCosaGrepeable:
    def test_los_tres_origenes_son_distintos(self):
        # Si dos colapsaran al mismo valor, el rastro dejaría de distinguir
        # justo lo que se lo pidió distinguir.
        assert len(rechazos.ORIGENES) == 3

    def test_un_pedido_que_pasa_los_tres_portones_no_registra_ninguno(self, env, caplog):
        # La contracara: un pedido que se atiende no deja rastro de rechazo.
        # Sin esto, «no hay rechazos en el log» no probaría nada.
        service, router, factory = env
        _con_una_corrida(service, factory)
        router.next = RoutedRequest(
            intent=Intent.QUERY_RESULTS,
            params={"formula": "Zr", "dato": "pasos_ionicos"},
        )
        with caplog.at_level(logging.INFO):
            reply = _mandar(service)
        assert _origenes(caplog) == []
        assert reply.ok

    def test_todos_los_origenes_declarados_se_usan_en_el_codigo(self):
        # Un origen que nadie registra es un portón que no existe, o uno que
        # se movió y dejó el nombre colgado.
        from pathlib import Path

        fuentes = "\n".join(
            p.read_text(encoding="utf-8")
            for p in Path("becario").rglob("*.py")
            if p.name != "rechazos.py"
        )
        for nombre in ("PRE_ROUTER", "ROUTER", "VOCABULARIO"):
            assert nombre in fuentes, f"{nombre} declarado pero nunca registrado"
