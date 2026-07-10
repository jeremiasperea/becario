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

    def job_state(self, job_id: JobId) -> Optional[str]:
        self.queried.append(job_id.value)
        return self.state


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


def _job(job_id="1", status=JobStatus.PENDING) -> TrackedJob:
    return TrackedJob(
        job_id=job_id, owner_id=ALICE.telegram_user_id, chat_id=999,
        ssh_user="alice", job_name="grafeno_dft", status=status,
    )


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
