"""Tests de infraestructura: SQLite real (en tmp), store con TTL, y
verificación de que los comandos SSH generados están correctamente quoteados.
"""
import json
import time

import pytest

from becario.domain.models import (
    ClusterIdentity,
    CommandResult,
    HistoryFilter,
    Intent,
    JobId,
    JobStatus,
    PendingAction,
    SlurmJobRequest,
    TrackedJob,
)
from becario.infrastructure.ssh_gateway import (
    SSHClusterGateway,
    SSHClusterGatewayFactory,
)
from becario.infrastructure.storage import (
    InMemoryConfirmationStore,
    SQLiteHistoryRepository,
    SQLiteJobTracker,
)
from becario.infrastructure.user_registry import JSONUserRegistry

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path):
    r = SQLiteHistoryRepository(str(tmp_path / "test.db"))
    r.ensure_schema()
    return r


class TestSQLiteHistory:
    def test_roundtrip(self, repo):
        repo.add(owner_id=1, job_id="100", nombre_trabajo="grafeno_dft", estado="COMPLETED")
        rows = repo.search(HistoryFilter(job_id="100"))
        assert len(rows) == 1
        assert rows[0]["nombre_trabajo"] == "grafeno_dft"

    def test_like_filter(self, repo):
        repo.add(owner_id=1, job_id="1", nombre_trabajo="grafeno_dft", estado="COMPLETED")
        repo.add(owner_id=1, job_id="2", nombre_trabajo="silicio_md", estado="RUNNING")
        rows = repo.search(HistoryFilter(name_contains="grafeno"))
        assert len(rows) == 1

    def test_sql_injection_is_inert(self, repo):
        repo.add(owner_id=1, job_id="1", nombre_trabajo="grafeno", estado="COMPLETED")
        # Si esto se interpolara, dropearía la tabla. Parametrizado, es texto.
        rows = repo.search(
            HistoryFilter(name_contains="'; DROP TABLE historial_calculos; --")
        )
        assert rows == []
        # La tabla sigue viva:
        assert len(repo.search(HistoryFilter(name_contains="grafeno"))) == 1

    def test_limit(self, repo):
        for i in range(10):
            repo.add(owner_id=1, job_id=str(i), nombre_trabajo=f"job_{i}", estado="COMPLETED")
        rows = repo.search(HistoryFilter(limit=3))
        assert len(rows) == 3

    def test_owner_isolation(self, repo):
        repo.add(owner_id=1, job_id="10", nombre_trabajo="job_alice", estado="COMPLETED")
        repo.add(owner_id=2, job_id="20", nombre_trabajo="job_bob", estado="COMPLETED")
        rows_alice = repo.search(HistoryFilter(owner_id=1))
        rows_bob = repo.search(HistoryFilter(owner_id=2))
        assert [r["nombre_trabajo"] for r in rows_alice] == ["job_alice"]
        assert [r["nombre_trabajo"] for r in rows_bob] == ["job_bob"]

    def test_owner_none_sees_everyone(self, repo):
        # Uso administrativo (no alcanzable desde Telegram): sin owner_id ve todo.
        repo.add(owner_id=1, job_id="10", nombre_trabajo="a", estado="COMPLETED")
        repo.add(owner_id=2, job_id="20", nombre_trabajo="b", estado="COMPLETED")
        rows = repo.search(HistoryFilter(owner_id=None, limit=50))
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# ConfirmationStore
# ---------------------------------------------------------------------------


def _action(requester_id: int = 1) -> PendingAction:
    return PendingAction(
        chat_id=1, requester_id=requester_id, intent=Intent.CANCEL_JOB,
        description="d", payload={"job_id": "1"},
    )


class TestConfirmationStore:
    def test_put_pop(self):
        store = InMemoryConfirmationStore(ttl_seconds=600)
        token = store.put(_action())
        assert store.pop(token) is not None
        assert store.pop(token) is None  # segundo pop: consumido

    def test_peek_does_not_consume(self):
        store = InMemoryConfirmationStore(ttl_seconds=600)
        token = store.put(_action())
        assert store.peek(token) is not None
        assert store.peek(token) is not None  # sigue ahí
        assert store.pop(token) is not None
        assert store.peek(token) is None  # ahora sí, consumido

    def test_ttl_expiry(self):
        store = InMemoryConfirmationStore(ttl_seconds=0.01)
        token = store.put(_action())
        time.sleep(0.05)
        assert store.pop(token) is None

    def test_peek_respects_ttl(self):
        store = InMemoryConfirmationStore(ttl_seconds=0.01)
        token = store.put(_action())
        time.sleep(0.05)
        assert store.peek(token) is None

    def test_purge(self):
        store = InMemoryConfirmationStore(ttl_seconds=0.01)
        store.put(_action())
        store.put(_action())
        time.sleep(0.05)
        assert store.purge_expired() == 2


# ---------------------------------------------------------------------------
# SSH command construction (sin conexión real: interceptamos _run)
# ---------------------------------------------------------------------------


class RecordingGateway(SSHClusterGateway):
    """Reemplaza la ejecución real para capturar el comando construido."""

    def __init__(self):
        super().__init__(host="fake", user="fake", key_path="/dev/null")
        self.commands: list[str] = []

    def _run(self, command: str) -> CommandResult:
        self.commands.append(command)
        return CommandResult(ok=True, stdout="ok")


class TestSSHCommandConstruction:
    def test_submit_uses_heredoc_and_validated_fields(self):
        gw = RecordingGateway()
        req = SlurmJobRequest(
            job_name="grafeno", partition="gpu", nodes=2,
            time_limit="02:00:00", script_path="/opt/calc.sh",
        )
        gw.submit_job(req)
        cmd = gw.commands[0]
        assert "sbatch <<'BECARIO_EOF'" in cmd
        assert "#SBATCH --job-name=grafeno" in cmd
        assert "#SBATCH --nodes=2" in cmd
        # cwd del job = dir del script: ahí caen slurm-%j.out y resultados.
        assert "#SBATCH --chdir=/opt" in cmd
        assert "bash /opt/calc.sh" in cmd

    def test_submit_parses_job_id_from_sbatch_output(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run",
            lambda cmd: CommandResult(ok=True, stdout="Submitted batch job 4242\n"),
        )
        result = gw.submit_job(SlurmJobRequest(script_path="/a/b.sh"))
        assert result.job_id == "4242"

    def test_submit_ok_but_unparsable_output_yields_no_job_id(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run", lambda cmd: CommandResult(ok=True, stdout="algo inesperado"),
        )
        result = gw.submit_job(SlurmJobRequest(script_path="/a/b.sh"))
        assert result.ok
        assert result.job_id is None

    def test_submit_failure_has_no_job_id(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run", lambda cmd: CommandResult(ok=False, stderr="sbatch: error"),
        )
        result = gw.submit_job(SlurmJobRequest(script_path="/a/b.sh"))
        assert not result.ok
        assert result.job_id is None

    def test_job_state_command_and_parsing(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run", lambda cmd: CommandResult(ok=True, stdout="COMPLETED\n"),
        )
        state = gw.job_state(JobId(value="4242"))
        assert state == "COMPLETED"

    def test_job_state_uses_parsable_format(self):
        gw = RecordingGateway()
        gw.job_state(JobId(value="99"))
        cmd = gw.commands[0]
        assert "--parsable2" in cmd
        assert "--noheader" in cmd
        assert "sacct -j 99" in cmd

    def test_job_state_returns_none_on_failure(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run", lambda cmd: CommandResult(ok=False, stderr="error"),
        )
        assert gw.job_state(JobId(value="1")) is None

    def test_job_state_returns_none_on_empty_output(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(gw, "_run", lambda cmd: CommandResult(ok=True, stdout=""))
        assert gw.job_state(JobId(value="1")) is None

    def test_cancel_is_quoted(self):
        gw = RecordingGateway()
        gw.cancel_job(JobId(value="12345"))
        assert gw.commands[0] == "scancel 12345"

    def test_status_without_job_id(self):
        gw = RecordingGateway()
        gw.job_status(None)
        assert "squeue" in gw.commands[0]

    def test_status_with_job_id(self):
        gw = RecordingGateway()
        gw.job_status(JobId(value="99"))
        assert "sacct -j 99" in gw.commands[0]

    def test_make_directory_quotes_path(self):
        gw = RecordingGateway()
        gw.make_directory("/home/user/calc dir")
        assert gw.commands[0] == "mkdir -p '/home/user/calc dir'"

    def test_make_directory_synthesizes_message_on_silent_success(self, monkeypatch):
        # mkdir -p exitoso no imprime nada: el gateway fabrica el mensaje.
        gw = RecordingGateway()
        monkeypatch.setattr(gw, "_run", lambda cmd: CommandResult(ok=True, stdout=""))
        result = gw.make_directory("/home/user/pruebas")
        assert result.ok
        assert result.stdout == "Directorio listo: /home/user/pruebas"

    def test_make_directory_failure_passes_through(self, monkeypatch):
        gw = RecordingGateway()
        failure = CommandResult(ok=False, stderr="mkdir: permiso denegado")
        monkeypatch.setattr(gw, "_run", lambda cmd: failure)
        result = gw.make_directory("/root/prohibido")
        # El fallo llega intacto: la rama del mensaje fabricado no se activa.
        assert result is failure

    def test_upload_creates_remote_dir_quoted(self, monkeypatch):
        gw = RecordingGateway()
        # Interceptamos SFTP: solo nos interesa el mkdir -p generado.
        class FakeSFTP:
            def put(self, local, remote): pass
            def close(self): pass
        class FakeClient:
            def open_sftp(self): return FakeSFTP()
        monkeypatch.setattr(gw, "_connection", lambda: FakeClient())
        result = gw.upload_file("/tmp/POSCAR.vasp", "/home/user/calc dir/POSCAR")
        assert result.ok
        assert gw.commands[0] == "mkdir -p '/home/user/calc dir'"


# ---------------------------------------------------------------------------
# JSONUserRegistry
# ---------------------------------------------------------------------------


class TestJSONUserRegistry:
    def test_missing_file_means_empty_roster(self, tmp_path):
        registry = JSONUserRegistry(str(tmp_path / "no_existe.json"))
        assert registry.get_identity(111) is None

    def test_loads_valid_entries(self, tmp_path):
        path = tmp_path / "users.json"
        path.write_text(json.dumps({
            "users": [
                {"telegram_user_id": 111, "ssh_user": "alice", "ssh_key_path": "/k/a",
                 "display_name": "Alice"},
                {"telegram_user_id": 222, "ssh_user": "bob", "ssh_key_path": "/k/b"},
            ]
        }))
        registry = JSONUserRegistry(str(path))
        alice = registry.get_identity(111)
        assert alice is not None
        assert alice.ssh_user == "alice"
        assert registry.get_identity(999) is None

    def test_invalid_entry_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "users.json"
        path.write_text(json.dumps({
            "users": [
                {"telegram_user_id": 111, "ssh_user": "BAD USER", "ssh_key_path": "/k/a"},
                {"telegram_user_id": 222, "ssh_user": "bob", "ssh_key_path": "/k/b"},
            ]
        }))
        registry = JSONUserRegistry(str(path))
        assert registry.get_identity(111) is None  # entrada inválida, ignorada
        assert registry.get_identity(222) is not None  # el resto del roster sigue

    def test_reload_picks_up_changes(self, tmp_path):
        path = tmp_path / "users.json"
        path.write_text(json.dumps({"users": []}))
        registry = JSONUserRegistry(str(path))
        assert registry.get_identity(111) is None
        path.write_text(json.dumps({
            "users": [{"telegram_user_id": 111, "ssh_user": "alice", "ssh_key_path": "/k/a"}]
        }))
        registry.reload()
        assert registry.get_identity(111) is not None


# ---------------------------------------------------------------------------
# SSHClusterGatewayFactory
# ---------------------------------------------------------------------------


class TestSSHClusterGatewayFactory:
    def test_same_user_returns_cached_instance(self):
        factory = SSHClusterGatewayFactory(default_host="cluster.edu.ar")
        identity = ClusterIdentity(telegram_user_id=1, ssh_user="alice", ssh_key_path="/k/a")
        gw1 = factory.for_identity(identity)
        gw2 = factory.for_identity(identity)
        assert gw1 is gw2

    def test_different_users_get_different_instances(self):
        factory = SSHClusterGatewayFactory(default_host="cluster.edu.ar")
        alice = ClusterIdentity(telegram_user_id=1, ssh_user="alice", ssh_key_path="/k/a")
        bob = ClusterIdentity(telegram_user_id=2, ssh_user="bob", ssh_key_path="/k/b")
        gw_alice = factory.for_identity(alice)
        gw_bob = factory.for_identity(bob)
        assert gw_alice is not gw_bob
        assert gw_alice._user == "alice"
        assert gw_bob._user == "bob"

    def test_per_identity_host_override(self):
        factory = SSHClusterGatewayFactory(default_host="cluster-general.edu.ar")
        identity = ClusterIdentity(
            telegram_user_id=1, ssh_user="alice", ssh_key_path="/k/a",
            ssh_host="cluster-especial.edu.ar",
        )
        gw = factory.for_identity(identity)
        assert gw._host == "cluster-especial.edu.ar"

    def test_falls_back_to_default_host(self):
        factory = SSHClusterGatewayFactory(default_host="cluster-general.edu.ar")
        identity = ClusterIdentity(telegram_user_id=1, ssh_user="alice", ssh_key_path="/k/a")
        gw = factory.for_identity(identity)
        assert gw._host == "cluster-general.edu.ar"


# ---------------------------------------------------------------------------
# SQLiteJobTracker
# ---------------------------------------------------------------------------


def _tracked_job(job_id="1", status=JobStatus.PENDING) -> TrackedJob:
    return TrackedJob(
        job_id=job_id, owner_id=111, chat_id=999, ssh_user="alice",
        job_name="grafeno_dft", status=status,
    )


class TestSQLiteJobTracker:
    def test_track_and_active_jobs_roundtrip(self, tmp_path):
        tracker = SQLiteJobTracker(str(tmp_path / "jobs.db"))
        tracker.track(_tracked_job("1"))
        active = tracker.active_jobs()
        assert len(active) == 1
        assert active[0].job_id == "1"
        assert active[0].status is JobStatus.PENDING
        assert not active[0].notified

    def test_update_status_persists(self, tmp_path):
        tracker = SQLiteJobTracker(str(tmp_path / "jobs.db"))
        tracker.track(_tracked_job("1"))
        tracker.update_status("1", 111, JobStatus.RUNNING)
        assert tracker.active_jobs()[0].status is JobStatus.RUNNING

    def test_mark_notified_removes_from_active(self, tmp_path):
        tracker = SQLiteJobTracker(str(tmp_path / "jobs.db"))
        tracker.track(_tracked_job("1"))
        tracker.mark_notified("1", 111)
        assert tracker.active_jobs() == []

    def test_isolated_by_owner(self, tmp_path):
        tracker = SQLiteJobTracker(str(tmp_path / "jobs.db"))
        tracker.track(TrackedJob(job_id="1", owner_id=111, chat_id=1, ssh_user="alice", job_name="a"))
        tracker.track(TrackedJob(job_id="1", owner_id=222, chat_id=2, ssh_user="bob", job_name="b"))
        # Mismo job_id, distinto owner: no se pisan (PK compuesta).
        assert len(tracker.active_jobs()) == 2
        tracker.mark_notified("1", 111)
        remaining = tracker.active_jobs()
        assert len(remaining) == 1
        assert remaining[0].owner_id == 222

    def test_track_upserts_existing_entry(self, tmp_path):
        tracker = SQLiteJobTracker(str(tmp_path / "jobs.db"))
        tracker.track(_tracked_job("1", status=JobStatus.PENDING))
        tracker.track(_tracked_job("1", status=JobStatus.RUNNING))  # re-track: no duplica
        active = tracker.active_jobs()
        assert len(active) == 1
        assert active[0].status is JobStatus.RUNNING

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "jobs.db")
        SQLiteJobTracker(path).track(_tracked_job("1"))
        reopened = SQLiteJobTracker(path)
        assert len(reopened.active_jobs()) == 1

    def test_workflow_roundtrip(self, tmp_path):
        tracker = SQLiteJobTracker(str(tmp_path / "jobs.db"))
        job = _tracked_job("1")
        job.workflow = "encut_scan"
        tracker.track(job)
        assert tracker.active_jobs()[0].workflow == "encut_scan"

    def test_calc_run_repository_roundtrip_and_scoping(self, tmp_path):
        from becario.infrastructure.storage import SQLiteCalcRunRepository

        repo = SQLiteCalcRunRepository(str(tmp_path / "runs.db"))
        repo.add(111, "1", "Zr_convergencia_encut", '{"encut": 400}', "/data/runs/a")
        repo.add(111, "2", "Zr_convergencia_encut", '{"encut": 500}', "/data/runs/b")
        repo.add(222, "3", "Zr_convergencia_encut", '{"encut": 400}', "/data/runs/c")

        rows = repo.find_by_name(111, "Zr_convergencia_encut")
        assert [r["job_id"] for r in rows] == ["2", "1"]  # más reciente primero
        assert all(r["owner_id"] == 111 for r in rows)  # nunca ve las de otro
        assert repo.find_by_name(111, "W_relajacion") == []

    def test_workflow_column_soft_migration(self, tmp_path):
        """Una base creada antes de la columna `workflow` se migra sola."""
        import sqlite3

        path = str(tmp_path / "jobs.db")
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE trabajos_monitoreados (
                    job_id TEXT NOT NULL,
                    owner_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    ssh_user TEXT NOT NULL,
                    job_name TEXT NOT NULL,
                    script_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    notified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (job_id, owner_id)
                )
                """
            )
            conn.execute(
                "INSERT INTO trabajos_monitoreados (job_id, owner_id, chat_id, ssh_user, job_name) "
                "VALUES ('1', 111, 999, 'alice', 'viejo')"
            )
        tracker = SQLiteJobTracker(path)
        jobs = tracker.active_jobs()
        assert len(jobs) == 1
        assert jobs[0].workflow == ""  # default para registros previos
