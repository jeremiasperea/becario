"""Tests del JobMonitorService: consulta trabajos activos, decide cuándo
notificar, y deja de rastrear una vez que avisó — sin Telegram real."""
from typing import Optional

import pytest

from becario.application.job_monitor import JobMonitorService
from becario.domain.models import ClusterIdentity, HistoryFilter, JobId, JobStatus, TrackedJob

ALICE = ClusterIdentity(telegram_user_id=111, ssh_user="alice", ssh_key_path="/k/a")


class FakeRegistry:
    def __init__(self, identities=None):
        if identities is None:
            identities = [ALICE]
        self._by_id = {i.telegram_user_id: i for i in identities}

    def get_identity(self, telegram_user_id: int):
        return self._by_id.get(telegram_user_id)


class FakeCluster:
    def __init__(self, state: Optional[str]):
        self.state = state
        self.queried: list[str] = []
        # Sistema de archivos remoto de mentira para la cosecha:
        self.remote_dirs: dict[str, list[str]] = {}
        self.remote_files: dict[str, str] = {}

    def job_state(self, job_id: JobId) -> Optional[str]:
        self.queried.append(job_id.value)
        return self.state

    def list_dir(self, remote_dir: str) -> Optional[list[str]]:
        return self.remote_dirs.get(remote_dir)

    def read_file(self, remote_path: str) -> Optional[str]:
        return self.remote_files.get(remote_path)


class FakeClusterFactory:
    def __init__(self, state: Optional[str] = "RUNNING"):
        self.cluster = FakeCluster(state)

    def for_identity(self, identity):
        return self.cluster


class FakeTracker:
    def __init__(self, jobs=None):
        self._jobs = list(jobs or [])
        self.status_updates: list[tuple[str, int, JobStatus]] = []
        self.notified_calls: list[tuple[str, int]] = []

    def track(self, job: TrackedJob) -> None:
        self._jobs.append(job)

    def active_jobs(self) -> list[TrackedJob]:
        return [j for j in self._jobs if not j.notified]

    def update_status(self, job_id, owner_id, status) -> None:
        self.status_updates.append((job_id, owner_id, status))
        for j in self._jobs:
            if j.job_id == job_id and j.owner_id == owner_id:
                j.status = status

    def mark_notified(self, job_id, owner_id) -> None:
        self.notified_calls.append((job_id, owner_id))
        for j in self._jobs:
            if j.job_id == job_id and j.owner_id == owner_id:
                j.notified = True


class FakeHistory:
    def __init__(self):
        self.added: list[dict] = []

    def add(self, owner_id, job_id, nombre_trabajo, estado) -> None:
        self.added.append({
            "owner_id": owner_id, "job_id": job_id,
            "nombre_trabajo": nombre_trabajo, "estado": estado,
        })

    def search(self, flt: HistoryFilter) -> list[dict]:
        return self.added


def _job(job_id="1", status=JobStatus.PENDING, workflow="", script_path="") -> TrackedJob:
    return TrackedJob(
        job_id=job_id, owner_id=ALICE.telegram_user_id, chat_id=999,
        ssh_user="alice", job_name="grafeno_dft", status=status,
        workflow=workflow, script_path=script_path,
    )


def _oszicar(energy: float) -> str:
    return (
        "       N       E                     dE\n"
        f"DAV:   1    {energy:.8E}    -0.1E-06\n"
        f"   1 F= {energy:.8E} E0= {energy:.8E}  d E =0.0\n"
    )


_POSCAR_ZR = "Zr2\n1.0\n3.23 0 0\n-1.6 2.8 0\n0 0 5.17\nZr\n2\nDirect\n0 0 0\n0.33 0.66 0.5\n"


def _scan_cluster(energies: dict[int, float]) -> "FakeClusterFactory":
    """Cluster con una corrida encut_scan terminada en /data/runs/zr."""
    factory = FakeClusterFactory("COMPLETED")
    cluster = factory.cluster
    run_dir = "/data/runs/zr"
    cluster.remote_dirs[run_dir] = ["run_vasp.sh", "POTCAR"] + [
        f"encut_{e}" for e in energies
    ]
    first = min(energies)
    cluster.remote_files[f"{run_dir}/encut_{first}/POSCAR"] = _POSCAR_ZR
    for encut, energy in energies.items():
        cluster.remote_files[f"{run_dir}/encut_{encut}/OSZICAR"] = _oszicar(energy)
    return factory


class TestJobMonitorService:
    def test_still_running_produces_no_notification(self):
        tracker = FakeTracker([_job(status=JobStatus.PENDING)])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("RUNNING"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert notes == []
        assert tracker.status_updates == [("1", ALICE.telegram_user_id, JobStatus.RUNNING)]
        assert tracker.notified_calls == []

    def test_completed_job_notifies_once_and_logs_history(self):
        tracker = FakeTracker([_job(status=JobStatus.RUNNING)])
        history = FakeHistory()
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("COMPLETED"),
            tracker=tracker, history=history,
        )
        notes = monitor.poll_and_notify()
        assert len(notes) == 1
        assert notes[0].chat_id == 999
        assert "terminó" in notes[0].text
        assert "1" in notes[0].text
        assert tracker.notified_calls == [("1", ALICE.telegram_user_id)]
        assert len(history.added) == 1
        assert history.added[0]["estado"] == "completado"

        # Una segunda vuelta no vuelve a notificar (ya no está en active_jobs).
        notes2 = monitor.poll_and_notify()
        assert notes2 == []

    def test_failed_job_uses_warning_icon(self):
        tracker = FakeTracker([_job(status=JobStatus.RUNNING)])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("FAILED"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert notes[0].text.startswith("⚠️")

    def test_cancelled_by_slurm_is_recognized(self):
        tracker = FakeTracker([_job(status=JobStatus.RUNNING)])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("CANCELLED by 111"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert len(notes) == 1
        assert tracker.status_updates[-1][2] is JobStatus.CANCELLED

    def test_unknown_state_keeps_polling_without_notifying(self):
        tracker = FakeTracker([_job(status=JobStatus.PENDING)])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory(None),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert notes == []
        assert tracker.status_updates == []  # no se pudo consultar, no se toca nada

    def test_deregistered_user_stops_tracking_without_notification(self):
        tracker = FakeTracker([_job()])
        monitor = JobMonitorService(
            registry=FakeRegistry(identities=[]),  # roster vacío: Alice ya no está
            cluster_factory=FakeClusterFactory("COMPLETED"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert notes == []  # nadie a quien avisarle
        assert tracker.active_jobs() == []  # pero se deja de rastrear igual

    def test_completed_encut_scan_adds_curve_notification(self):
        # 300 lejos, 350 a 0.4 meV/át del máximo (converge), 400 = referencia.
        factory = _scan_cluster({300: -16.9000, 350: -16.94920, 400: -16.9500})
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, workflow="encut_scan",
            script_path="/data/runs/zr/run_vasp.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=factory,
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert len(notes) == 2  # aviso de fin + curva
        curve = notes[1]
        assert curve.chat_id == 999
        assert curve.monospace
        assert "Convergencia de ENCUT" in curve.text
        assert "300" in curve.text and "400" in curve.text
        assert "ENCUT recomendado: 350 eV" in curve.text

    def test_scan_without_converged_point_warns(self):
        factory = _scan_cluster({300: -16.60, 350: -16.80, 400: -16.9500})
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, workflow="encut_scan",
            script_path="/data/runs/zr/run_vasp.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=factory,
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert "extender el barrido" in notes[1].text

    def test_failed_scan_does_not_harvest(self):
        factory = _scan_cluster({300: -16.90, 400: -16.95})
        factory.cluster.state = "FAILED"
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, workflow="encut_scan",
            script_path="/data/runs/zr/run_vasp.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=factory,
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert len(notes) == 1  # solo el aviso de que falló

    def test_scan_with_unreadable_run_dir_warns(self):
        factory = FakeClusterFactory("COMPLETED")  # sin remote_dirs
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, workflow="encut_scan",
            script_path="/data/runs/zr/run_vasp.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=factory,
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert len(notes) == 2
        assert "No pude leer" in notes[1].text

    def test_failed_job_without_remote_logs_explains_probable_cause(self):
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, script_path="/root/runs/x/run_vasp.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("FAILED"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert "🔍 Diagnóstico" in notes[0].text
        assert "nunca llegara a ejecutarse" in notes[0].text

    def test_failed_job_includes_log_tails(self):
        factory = FakeClusterFactory("FAILED")
        factory.cluster.remote_files["/data/runs/x/slurm-1.out"] = (
            "bash: error de sintaxis\ncp: cannot stat 'POTCAR'\n"
        )
        factory.cluster.remote_files["/data/runs/x/vasp.out"] = (
            "LAPACK: Routine ZPOTRF failed!\n"
        )
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, script_path="/data/runs/x/run_vasp.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=factory,
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        text = notes[0].text
        assert "cp: cannot stat 'POTCAR'" in text
        assert "ZPOTRF" in text

    def test_failed_scan_shows_vasp_out_of_last_started_point(self):
        factory = _scan_cluster({300: -16.90})  # solo el primer punto corrió
        factory.cluster.state = "FAILED"
        factory.cluster.remote_files["/data/runs/zr/encut_300/vasp.out"] = (
            "internal error in SETUP_DEG_CLUSTERS\n"
        )
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, workflow="encut_scan",
            script_path="/data/runs/zr/run_vasp.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=factory,
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert "encut_300/vasp.out" in notes[0].text
        assert "SETUP_DEG_CLUSTERS" in notes[0].text

    def test_completed_job_has_no_diagnostics(self):
        tracker = FakeTracker([_job(
            status=JobStatus.RUNNING, script_path="/data/runs/x/run.sh",
        )])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("COMPLETED"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert "Diagnóstico" not in notes[0].text

    def test_plain_job_never_harvests(self):
        tracker = FakeTracker([_job(status=JobStatus.RUNNING,
                                    script_path="/data/runs/x/calc.sh")])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("COMPLETED"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert len(notes) == 1

    def test_multiple_active_jobs_are_all_checked(self):
        tracker = FakeTracker([
            _job(job_id="1", status=JobStatus.RUNNING),
            _job(job_id="2", status=JobStatus.RUNNING),
        ])
        monitor = JobMonitorService(
            registry=FakeRegistry(), cluster_factory=FakeClusterFactory("COMPLETED"),
            tracker=tracker, history=FakeHistory(),
        )
        notes = monitor.poll_and_notify()
        assert len(notes) == 2
        assert {n.text[:1] for n in notes} == {"✅"}
