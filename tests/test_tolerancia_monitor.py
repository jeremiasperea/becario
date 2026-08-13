"""El cierre del loop: acuse de entrega (T4) y trabajos perdidos (T6).

Parte de la batería de fallos inyectados de `docs/plan_tolerancia_fallos.md`.
El resto de la suite prueba que el sistema FUNCIONA; esta prueba que AGUANTA,
que es otra cosa: acá todos los dobles fallan a propósito. Ninguno de estos
tests necesita cluster, ni Ollama, ni red — y esa es la razón de que puedan
existir, y también de que no existieran: las fallas que cubren no se
reproducen pidiéndole cosas a un sistema sano.
"""
import sqlite3
from becario.application.job_monitor import (
    JobMonitorService,
    _MAX_CONSULTAS_SIN_RESPUESTA,
)
from becario.domain.models import JobStatus, TrackedJob
from becario.infrastructure.storage import SQLiteJobTracker
from .test_job_monitor import (
    FakeClusterFactory as MonClusterFactory,
    FakeHistory as MonHistory,
    FakeRegistry as MonRegistry,
    FakeTracker as MonTracker,
    _job as _mon_job,
)



# ---------------------------------------------------------------------------
# T6 — un trabajo del que no hay noticias no se persigue para siempre
# ---------------------------------------------------------------------------


def _monitor_ciego(tracker):
    """Monitor cuyo cluster nunca puede leer el estado (`job_state -> None`),
    que es lo que pasa cuando Slurm ya purgó el trabajo de `sacct`."""
    return JobMonitorService(
        registry=MonRegistry(),
        cluster_factory=MonClusterFactory(None),
        tracker=tracker,
        history=MonHistory(),
    )


class TestTrabajosPerdidos:
    def test_tras_la_racha_se_avisa_una_vez_y_se_suelta(self):
        job = _mon_job(status=JobStatus.RUNNING)
        job.poll_attempts = _MAX_CONSULTAS_SIN_RESPUESTA - 1
        tracker = MonTracker([job])
        monitor = _monitor_ciego(tracker)

        notas = monitor.poll_and_notify()

        assert len(notas) == 1
        assert "Perdí el rastro" in notas[0].text
        assert notas[0].acuse is not None
        # Sigue activo hasta que el aviso salga (mismo contrato que T4).
        assert tracker.active_jobs()
        monitor.confirm_delivery(notas[0].acuse)
        assert tracker.active_jobs() == []
        # Y no vuelve a avisar nunca más.
        assert monitor.poll_and_notify() == []

    def test_antes_de_la_racha_solo_reintenta(self):
        tracker = MonTracker([_mon_job(status=JobStatus.RUNNING)])
        monitor = _monitor_ciego(tracker)

        assert monitor.poll_and_notify() == []
        assert tracker.active_jobs()[0].poll_attempts == 1
        assert tracker.active_jobs(), "se soltó al primer fallo"

    def test_una_lectura_buena_corta_la_racha(self):
        """Lo que hace que la cuenta sea de consultas SEGUIDAS y no
        acumuladas: un SSH que se cayó un rato no puede acercar un trabajo
        sano a darse por perdido."""
        job = _mon_job(status=JobStatus.RUNNING)
        job.poll_attempts = _MAX_CONSULTAS_SIN_RESPUESTA - 1
        tracker = MonTracker([job])
        factory = MonClusterFactory("RUNNING")  # el cluster vuelve a contestar
        monitor = JobMonitorService(
            registry=MonRegistry(), cluster_factory=factory,
            tracker=tracker, history=MonHistory(),
        )

        assert monitor.poll_and_notify() == []
        assert tracker.active_jobs()[0].poll_attempts == 0, "la racha no se reinició"


class TestTrackerCuentaRachas:
    def test_record_y_clear_sobre_sqlite(self, tmp_path):
        tracker = SQLiteJobTracker(str(tmp_path / "b.db"))
        tracker.track(TrackedJob(
            job_id="7", owner_id=1, chat_id=1, ssh_user="a", job_name="zr",
        ))

        assert tracker.record_unreachable("7", 1) == 1
        assert tracker.record_unreachable("7", 1) == 2
        assert tracker.active_jobs()[0].poll_attempts == 2

        tracker.clear_unreachable("7", 1)
        assert tracker.active_jobs()[0].poll_attempts == 0

    def test_una_base_vieja_se_migra_sin_perder_los_trabajos(self, tmp_path):
        """La columna se agrega a bases que ya existen, como se hizo antes
        con `script_path` y `workflow`."""
        db = tmp_path / "vieja.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                CREATE TABLE trabajos_monitoreados (
                    job_id TEXT NOT NULL, owner_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL, ssh_user TEXT NOT NULL,
                    job_name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING',
                    notified INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (job_id, owner_id)
                )
                """
            )
            conn.execute(
                "INSERT INTO trabajos_monitoreados "
                "(job_id, owner_id, chat_id, ssh_user, job_name) VALUES ('9',1,1,'a','viejo')"
            )

        tracker = SQLiteJobTracker(str(db))

        activos = tracker.active_jobs()
        assert [j.job_id for j in activos] == ["9"]
        assert activos[0].poll_attempts == 0
        assert tracker.record_unreachable("9", 1) == 1
