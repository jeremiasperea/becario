"""Lo transitorio se distingue de lo definitivo, y solo se reintenta lo
que es seguro repetir (Etapa 2).

Parte de la batería de fallos inyectados de `docs/plan_tolerancia_fallos.md`.
El resto de la suite prueba que el sistema FUNCIONA; esta prueba que AGUANTA,
que es otra cosa: acá todos los dobles fallan a propósito. Ninguno de estos
tests necesita cluster, ni Ollama, ni red — y esa es la razón de que puedan
existir, y también de que no existieran: las fallas que cubren no se
reproducen pidiéndole cosas a un sistema sano.
"""
import pytest
from becario.domain.models import (
    CommandFailureReason,
    CommandResult,
    JobId,
    RouterFailureReason,
    RouterUnavailableError,
    SlurmJobRequest,
)
from becario.infrastructure.ollama_router import OllamaRouter
from becario.infrastructure.ssh_gateway import SSHClusterGateway
from becario.reintentos import espera_con_jitter, reintentar
from becario.application.job_monitor import _MAX_CONSULTAS_SIN_RESPUESTA, JobMonitorService
from becario.domain.models import JobStatus
from .test_job_monitor import FakeClusterFactory as MonClusterFactory
from .test_job_monitor import FakeHistory as MonHistory
from .test_job_monitor import FakeRegistry as MonRegistry
from .test_job_monitor import FakeTracker as MonTracker
from .test_job_monitor import _job as _mon_job



# ---------------------------------------------------------------------------
# Etapa 2 — distinguir lo transitorio de lo definitivo, y reintentar
# ---------------------------------------------------------------------------


class TestTercerEstado:
    def test_un_comando_que_fallo_no_es_transitorio(self):
        # `permission denied` es igual de denegado la segunda vez.
        r = CommandResult(ok=False, stderr="denied", reason=CommandFailureReason.COMMAND)
        assert r.transitorio is False

    @pytest.mark.parametrize(
        "reason", [CommandFailureReason.TRANSPORT, CommandFailureReason.TIMEOUT]
    )
    def test_no_haber_podido_hablar_si_lo_es(self, reason):
        assert CommandResult(ok=False, reason=reason).transitorio is True

    def test_el_exito_no_es_transitorio(self):
        assert CommandResult(ok=True).transitorio is False


class TestReintentar:
    def _pausas(self):
        pausas: list[float] = []
        return pausas, pausas.append

    def test_devuelve_al_primer_intento_bueno(self):
        llamadas = []

        def operacion():
            llamadas.append(1)
            return CommandResult(ok=True)

        r = reintentar(operacion, resultado_transitorio=lambda x: x.transitorio)
        assert r.ok is True
        assert len(llamadas) == 1

    def test_reintenta_lo_transitorio_y_se_rinde_al_tope(self):
        pausas, dormir = self._pausas()
        llamadas = []

        def operacion():
            llamadas.append(1)
            return CommandResult(ok=False, reason=CommandFailureReason.TRANSPORT)

        r = reintentar(
            operacion, intentos=3,
            resultado_transitorio=lambda x: x.transitorio,
            dormir=dormir, azar=lambda: 1.0,
        )
        assert len(llamadas) == 3
        assert r.ok is False
        # Se duerme entre intentos, no después del último.
        assert len(pausas) == 2

    def test_no_reintenta_lo_definitivo(self):
        llamadas = []

        def operacion():
            llamadas.append(1)
            return CommandResult(ok=False, reason=CommandFailureReason.COMMAND)

        reintentar(
            operacion, intentos=3, resultado_transitorio=lambda x: x.transitorio
        )
        assert len(llamadas) == 1, "reintentó algo que nunca iba a andar"

    def test_el_backoff_crece_y_respeta_el_tope(self):
        # `azar=1.0` saca el jitter del medio para poder ver la progresión.
        crudas = [
            espera_con_jitter(i, base=0.5, tope=2.0, azar=lambda: 1.0)
            for i in range(1, 5)
        ]
        assert crudas == [0.5, 1.0, 2.0, 2.0]

    def test_el_jitter_reparte_la_espera(self):
        # Sin jitter, N clientes que fallan por lo mismo vuelven todos
        # juntos en el mismo instante.
        assert espera_con_jitter(3, base=1.0, tope=8.0, azar=lambda: 0.0) == 0.0
        assert espera_con_jitter(3, base=1.0, tope=8.0, azar=lambda: 0.5) == 2.0

    def test_las_excepciones_transitorias_se_reintentan(self):
        llamadas = []

        def operacion():
            llamadas.append(1)
            raise RouterUnavailableError(RouterFailureReason.UNREACHABLE)

        with pytest.raises(RouterUnavailableError):
            reintentar(
                operacion, intentos=3,
                excepcion_transitoria=lambda e: True,
                dormir=lambda _s: None,
            )
        assert len(llamadas) == 3


class TestQueSeReintentaYQueNo:
    """La parte peligrosa: reintentar lo que no es idempotente duplica
    trabajo real en el cluster."""

    def _gateway_que_falla(self):
        intentos = []

        gw = SSHClusterGateway(host="h", user="u", key_path="/dev/null")
        def _run_once(command):
            intentos.append(command)
            return CommandResult(
                ok=False, stderr="sin red", reason=CommandFailureReason.TRANSPORT
            )
        gw._run_once = _run_once
        return gw, intentos

    def test_sbatch_NO_se_reintenta(self):
        """El más importante del archivo.

        Si el `sbatch` llega al cluster y la respuesta se pierde, reintentar
        encola el trabajo dos veces: el usuario no ve el primero (no
        tenemos su id) y paga las horas igual.
        """
        gw, intentos = self._gateway_que_falla()
        gw.submit_job(SlurmJobRequest(script_path="/a/b.sh"))
        assert len(intentos) == 1, "se reintentó un envío: puede duplicar el trabajo"

    def test_las_consultas_si_se_reintentan(self):
        gw, intentos = self._gateway_que_falla()
        gw.job_state(JobId(value="42"))
        assert len(intentos) == 3

    def test_mkdir_se_reintenta_por_idempotente(self):
        gw, intentos = self._gateway_que_falla()
        gw.make_directory("/data/x")
        assert len(intentos) == 3

    def test_scancel_se_reintenta_por_idempotente(self):
        # Cancelar dos veces algo ya cancelado no hace nada; crear no.
        gw, intentos = self._gateway_que_falla()
        gw.cancel_job(JobId(value="42"))
        assert len(intentos) == 3


class TestPoliticaDeReintentoDeOllama:
    def _router_que_falla(self, excepcion):
        router = OllamaRouter(base_url="http://x", model="m", timeout=180)
        intentos = []

        def _chat_once(system_prompt, user_text, schema):
            intentos.append(1)
            raise excepcion

        router._chat_once = _chat_once
        return router, intentos

    def test_el_timeout_NO_se_reintenta(self):
        """Ya se esperaron 180 s: repetir le suma otros 180 al usuario para
        llegar casi seguro al mismo lugar."""
        router, intentos = self._router_que_falla(
            RouterUnavailableError(RouterFailureReason.TIMEOUT, timeout_seconds=180)
        )
        with pytest.raises(RouterUnavailableError):
            router.route("relajá el ZrO2")
        assert len(intentos) == 1

    def test_el_servidor_caido_si_se_reintenta(self):
        # Falla en milisegundos y se arregla solo bastante seguido.
        router, intentos = self._router_que_falla(
            RouterUnavailableError(RouterFailureReason.UNREACHABLE)
        )
        with pytest.raises(RouterUnavailableError):
            router.route("x")
        assert len(intentos) == 3


class TestMonitorDistingueNoAlcanzable:
    def test_el_cluster_caido_no_cuenta_para_la_racha(self):
        """El pago de haber partido `ok=False` en dos: un rato de red mala
        ya no acerca a un trabajo SANO a que se lo dé por perdido."""
        job = _mon_job(status=JobStatus.RUNNING)
        job.poll_attempts = _MAX_CONSULTAS_SIN_RESPUESTA - 1
        tracker = MonTracker([job])
        monitor = JobMonitorService(
            registry=MonRegistry(),
            cluster_factory=MonClusterFactory(None, reachable=False),
            tracker=tracker, history=MonHistory(),
        )

        assert monitor.poll_and_notify() == []
        assert tracker.active_jobs()[0].poll_attempts == (
            _MAX_CONSULTAS_SIN_RESPUESTA - 1
        ), "un cluster caído sumó a la racha del trabajo"

    def test_sacct_que_contesta_sin_conocerlo_si_cuenta(self):
        job = _mon_job(status=JobStatus.RUNNING)
        job.poll_attempts = _MAX_CONSULTAS_SIN_RESPUESTA - 1
        tracker = MonTracker([job])
        monitor = JobMonitorService(
            registry=MonRegistry(),
            cluster_factory=MonClusterFactory(None, reachable=True),
            tracker=tracker, history=MonHistory(),
        )

        notas = monitor.poll_and_notify()
        assert len(notas) == 1
        assert "Perdí el rastro" in notas[0].text
