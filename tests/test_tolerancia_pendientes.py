"""Nada queda huérfano en el cluster (T7) y el cierre es ordenado (T8).

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
# T7 — nada queda huérfano en el cluster
# ---------------------------------------------------------------------------


class TestGuardDelBorrado:
    """El `rm -rf` es la única operación destructiva del gateway. El guard
    va con la misma lógica que el `shlex.quote`: la ruta la arma la
    aplicación, no el usuario — pero se valida igual."""

    def _gw(self):
        gw = SSHClusterGateway(host="h", user="u", key_path="/dev/null")
        corridos = []
        gw._run_once = lambda cmd: (
            corridos.append(cmd), CommandResult(ok=True)
        )[1]
        return gw, corridos

    @pytest.mark.parametrize(
        "ruta",
        [
            "/data/runs/W_relajacion",          # una corrida de verdad
            "/data/runs/.pending",              # el contenedor: se llevaría todo
            "/",                                # …
            "relativa/.pending/x",              # no absoluta
            "/data/runs/.pending/../../etc",    # escapa con ..
            "",
        ],
    )
    def test_se_niega_fuera_del_area_de_pendientes(self, ruta, caplog):
        gw, corridos = self._gw()
        with caplog.at_level(logging.ERROR):
            r = gw.discard_pending(ruta)
        assert r.ok is False
        assert corridos == [], f"ejecutó un rm sobre {ruta!r}"

    def test_borra_una_corrida_pendiente_de_verdad(self):
        gw, corridos = self._gw()
        r = gw.discard_pending("/data/runs/.pending/W_relajacion_x")
        assert r.ok is True
        assert len(corridos) == 1
        assert "rm -rf" in corridos[0]
        assert "/data/runs/.pending/W_relajacion_x" in corridos[0]

    @pytest.mark.parametrize(
        "base", ["/data/runs", "/data/runs/.pending/x", "relativa/.pending"]
    )
    def test_el_barrido_solo_acepta_el_contenedor(self, base):
        gw, corridos = self._gw()
        assert gw.sweep_pending(base, 120).ok is False
        assert corridos == []

    def test_el_barrido_toca_solo_los_hijos_directos(self):
        gw, corridos = self._gw()
        gw.sweep_pending("/data/runs/.pending", 120)
        cmd = corridos[0]
        assert "-mindepth 1 -maxdepth 1" in cmd, "podría borrar el contenedor"
        assert "-mmin +120" in cmd


class TestCorridaSinConfirmarNoQuedaSuelta:
    def _preparar(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={"formula": "W", "tipo_calculo": "relajacion"},
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="relajá W"
        )
        return service, factory.gateways["alice"], reply.confirmation_token

    def test_al_confirmar_se_mueve_al_lugar_definitivo(self, env):
        service, cluster, token = self._preparar(env)

        service.confirm(token, requester_id=ALICE.telegram_user_id)

        assert len(cluster.moved) == 1
        origen, destino = cluster.moved[0]
        assert ".pending/" in origen
        assert ".pending/" not in destino
        assert cluster.submitted, "no se envió"

    def test_si_el_mv_falla_no_se_envia_nada(self, env):
        """El `mv` va ANTES del `sbatch`: si falla, no puede quedar un
        trabajo encolado apuntando a un directorio que no existe."""
        service, cluster, token = self._preparar(env)
        cluster.move_run_result = CommandResult(
            ok=False, stderr="no space left", reason=CommandFailureReason.COMMAND
        )

        reply = service.confirm(token, requester_id=ALICE.telegram_user_id)

        assert cluster.submitted == [], "envió con la corrida sin mover"
        assert "No envié nada" in reply.text

    def test_al_cancelar_se_borra_del_cluster(self, env):
        service, cluster, token = self._preparar(env)

        service.reject(token, requester_id=ALICE.telegram_user_id)

        assert len(cluster.discarded) == 1
        assert ".pending/" in cluster.discarded[0]
        assert cluster.moved == []

    def test_un_borrado_fallido_no_rompe_la_cancelacion(self, env):
        # Cancelar es lo que el usuario pidió y ya está hecho. Si la
        # limpieza falla queda el barrido por edad.
        service, cluster, token = self._preparar(env)
        cluster.discard_pending = lambda path: (_ for _ in ()).throw(
            RuntimeError("sin red")
        )

        reply = service.reject(token, requester_id=ALICE.telegram_user_id)

        assert "cancelada" in reply.text

    def test_preparar_barre_lo_viejo_de_esa_cuenta(self, env):
        """La red que cubre lo que el borrado explícito no puede: nadie
        aprieta ❌ cuando deja vencer una confirmación."""
        _, cluster, _ = self._preparar(env)

        assert len(cluster.swept) == 1
        base, minutos = cluster.swept[0]
        assert base.endswith("/.pending")
        assert minutos > 30, "barre antes de que venza un pedido pendiente"
