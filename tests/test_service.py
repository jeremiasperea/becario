"""Tests del BecarioService con dobles de prueba.

Gracias a los puertos (Protocols), el servicio se testea completo sin
Telegram, sin SSH y sin Ollama. Estos tests son también la especificación
viva del modelo multiusuario: cada persona opera con su propia identidad
y conexión, sin rol admin, con aislamiento verificado explícitamente.
"""
from typing import Optional

import pytest

from becario.application.services import (
    NOT_REGISTERED_TEXT,
    BecarioService,
    HELP_TEXT,
)
from becario.domain.models import (
    ClusterIdentity,
    CommandResult,
    HistoryFilter,
    Intent,
    JobId,
    JobStatus,
    RoutedRequest,
    SlurmJobRequest,
    StructureRequest,
    StructureResult,
    TrackedJob,
)
from becario.infrastructure.storage import InMemoryConfirmationStore

# ---------------------------------------------------------------------------
# Identidades de prueba
# ---------------------------------------------------------------------------

ALICE = ClusterIdentity(telegram_user_id=111, ssh_user="alice", ssh_key_path="/k/a", display_name="Alice")
BOB = ClusterIdentity(telegram_user_id=222, ssh_user="bob", ssh_key_path="/k/b", display_name="Bob")

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRouter:
    def __init__(self):
        self.next: RoutedRequest = RoutedRequest(intent=Intent.UNKNOWN, params={})

    def route(self, user_text: str) -> RoutedRequest:
        return self.next


class FakeUserRegistry:
    def __init__(self, identities: Optional[list[ClusterIdentity]] = None):
        self._by_id = {i.telegram_user_id: i for i in (identities or [])}

    def get_identity(self, telegram_user_id: int) -> Optional[ClusterIdentity]:
        return self._by_id.get(telegram_user_id)


class FakeCluster:
    """Una instancia por usuario — así los tests pueden verificar que
    nunca se cruzan comandos entre cuentas."""

    def __init__(self, ssh_user: str):
        self.ssh_user = ssh_user
        self.submitted: list[SlurmJobRequest] = []
        self.cancelled: list[JobId] = []
        self.status_calls: list[Optional[JobId]] = []
        self.uploads: list[tuple[str, str]] = []

    def submit_job(self, req: SlurmJobRequest) -> CommandResult:
        self.submitted.append(req)
        return CommandResult(ok=True, stdout="Submitted batch job 4242", job_id="4242")

    def cancel_job(self, job_id: JobId) -> CommandResult:
        self.cancelled.append(job_id)
        return CommandResult(ok=True, stdout=f"Job {job_id} cancelado")

    def job_status(self, job_id: Optional[JobId]) -> CommandResult:
        self.status_calls.append(job_id)
        return CommandResult(ok=True, stdout="JOBID  NAME  STATE")

    def upload_file(self, local_path: str, remote_path: str) -> CommandResult:
        self.uploads.append((local_path, remote_path))
        return CommandResult(ok=True, stdout=f"Archivo subido a {remote_path}")

    def job_state(self, job_id: JobId) -> str:
        return "RUNNING"


class FakeClusterGatewayFactory:
    def __init__(self):
        self.gateways: dict[str, FakeCluster] = {}

    def for_identity(self, identity: ClusterIdentity) -> FakeCluster:
        if identity.ssh_user not in self.gateways:
            self.gateways[identity.ssh_user] = FakeCluster(identity.ssh_user)
        return self.gateways[identity.ssh_user]


class FakeStructureBuilder:
    def __init__(self):
        self.requests: list[StructureRequest] = []

    def build(self, req: StructureRequest) -> StructureResult:
        self.requests.append(req)
        return StructureResult(
            local_path="/tmp/POSCAR_test.vasp",
            filename="POSCAR_test.vasp",
            chemical_formula=req.formula,
            n_atoms=2,
            cell_summary="a=5.430 Å, b=5.430 Å, c=5.430 Å",
        )


class FakeJobTracker:
    def __init__(self):
        self.tracked: list[TrackedJob] = []
        self.notified_calls: list[tuple[str, int]] = []
        self.status_updates: list[tuple[str, int, JobStatus]] = []

    def track(self, job: TrackedJob) -> None:
        self.tracked.append(job)

    def active_jobs(self) -> list[TrackedJob]:
        return [j for j in self.tracked if not j.notified]

    def update_status(self, job_id: str, owner_id: int, status: JobStatus) -> None:
        self.status_updates.append((job_id, owner_id, status))

    def mark_notified(self, job_id: str, owner_id: int) -> None:
        self.notified_calls.append((job_id, owner_id))
        for j in self.tracked:
            if j.job_id == job_id and j.owner_id == owner_id:
                j.notified = True


class FakeHistory:
    def __init__(self, rows: Optional[list[dict]] = None):
        self.rows = rows or []
        self.last_filter: Optional[HistoryFilter] = None
        self.added: list[dict] = []

    def add(self, owner_id: int, job_id: str, nombre_trabajo: str, estado: str) -> None:
        self.added.append({
            "owner_id": owner_id, "job_id": job_id,
            "nombre_trabajo": nombre_trabajo, "estado": estado,
        })

    def search(self, flt: HistoryFilter) -> list[dict]:
        self.last_filter = flt
        return self.rows


@pytest.fixture()
def env():
    router = FakeRouter()
    registry = FakeUserRegistry([ALICE, BOB])
    cluster_factory = FakeClusterGatewayFactory()
    history = FakeHistory()
    confirmations = InMemoryConfirmationStore(ttl_seconds=600)
    structures = FakeStructureBuilder()
    job_tracker = FakeJobTracker()
    service = BecarioService(
        router=router,
        registry=registry,
        cluster_factory=cluster_factory,
        history=history,
        confirmations=confirmations,
        structures=structures,
        job_tracker=job_tracker,
    )
    return service, router, cluster_factory, history, confirmations, structures, job_tracker


# ---------------------------------------------------------------------------
# Registro / identidad
# ---------------------------------------------------------------------------


class TestUserRegistration:
    def test_unregistered_user_is_rejected(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={})
        reply = service.handle_text(chat_id=999, user_id=999, text="estado")
        assert reply.text == NOT_REGISTERED_TEXT

    def test_registered_user_proceeds(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={})
        reply = service.handle_text(chat_id=111, user_id=ALICE.telegram_user_id, text="estado")
        assert reply.text != NOT_REGISTERED_TEXT


class TestMultiUserIsolation:
    def test_each_user_gets_own_cluster_connection(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={})
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="estado")
        service.handle_text(chat_id=2, user_id=BOB.telegram_user_id, text="estado")
        assert set(factory.gateways.keys()) == {"alice", "bob"}
        assert len(factory.gateways["alice"].status_calls) == 1
        assert len(factory.gateways["bob"].status_calls) == 1

    def test_submit_runs_on_requesters_own_account(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
        )
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert len(factory.gateways["alice"].submitted) == 1
        assert "bob" not in factory.gateways  # nunca se tocó la cuenta de Bob

    def test_history_is_scoped_to_owner(self, env):
        service, router, _, history, *_ = env
        router.next = RoutedRequest(intent=Intent.QUERY_DB, params={})
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="historial")
        assert history.last_filter.owner_id == ALICE.telegram_user_id
        service.handle_text(chat_id=2, user_id=BOB.telegram_user_id, text="historial")
        assert history.last_filter.owner_id == BOB.telegram_user_id


# ---------------------------------------------------------------------------
# Flujo de confirmación
# ---------------------------------------------------------------------------


class TestConfirmationFlow:
    def test_submit_requires_confirmation_and_does_not_execute(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM,
            params={"nombre_trabajo": "grafeno", "script_remoto": "/opt/calc.sh"},
        )
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="corré grafeno")
        assert reply.needs_confirmation
        assert reply.confirmation_token
        assert factory.gateways["alice"].submitted == []  # ¡no se ejecutó!

    def test_confirm_executes_exactly_once(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
        )
        token = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="x"
        ).confirmation_token

        first = service.confirm(token, requester_id=ALICE.telegram_user_id)
        assert "4242" in first.text
        assert len(factory.gateways["alice"].submitted) == 1

        second = service.confirm(token, requester_id=ALICE.telegram_user_id)
        assert "expiró" in second.text
        assert len(factory.gateways["alice"].submitted) == 1

    def test_reject_never_executes(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "12345"})
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        reply = service.reject(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert "cancelada" in reply.text.lower()
        assert factory.gateways["alice"].cancelled == []

    def test_cancel_flow_end_to_end(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "777"})
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá el 777")
        assert prep.needs_confirmation
        assert "777" in prep.text
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert [j.value for j in factory.gateways["alice"].cancelled] == ["777"]

    def test_expired_or_unknown_token(self, env):
        service, *_ = env
        reply = service.confirm("token_inexistente", requester_id=ALICE.telegram_user_id)
        assert "expiró" in reply.text

    def test_foreign_confirmation_is_rejected(self, env):
        """Bob no puede confirmar una acción que pidió Alice."""
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "777"})
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá el 777")

        reply = service.confirm(prep.confirmation_token, requester_id=BOB.telegram_user_id)
        assert "no te pertenece" in reply.text.lower()
        assert factory.gateways["alice"].cancelled == []
        assert "bob" not in factory.gateways

        # La confirmación de Alice sigue viva pese al intento ajeno:
        reply2 = service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert "cancelado" in reply2.text.lower() or "4242" in reply2.text

    def test_foreign_rejection_is_also_blocked(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "777"})
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá el 777")
        reply = service.reject(prep.confirmation_token, requester_id=BOB.telegram_user_id)
        assert "no te pertenece" in reply.text.lower()


class TestDestructiveInputValidation:
    def test_submit_with_injection_is_rejected_before_confirmation(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM,
            params={"script_remoto": "/tmp/x.sh; rm -rf /"},
        )
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert not reply.needs_confirmation
        assert "inválido" in reply.text.lower() or "Parámetros" in reply.text
        assert factory.gateways["alice"].submitted == []

    def test_cancel_without_job_id(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá")
        assert not reply.needs_confirmation
        assert "job_id" in reply.text
        assert factory.gateways["alice"].cancelled == []

    def test_cancel_with_evil_job_id(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.CANCEL_JOB, params={"job_id": "1; scancel -u root"}
        )
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert not reply.needs_confirmation
        assert factory.gateways["alice"].cancelled == []


class TestReadOnlyActions:
    def test_status_executes_directly(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="estado")
        assert not reply.needs_confirmation
        assert "Estado" in reply.text
        assert factory.gateways["alice"].status_calls == [None]

    def test_status_with_job_id(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={"job_id": "555"})
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="estado del 555")
        assert factory.gateways["alice"].status_calls[0].value == "555"

    def test_history_with_results(self, env):
        service, router, _, history, *_ = env
        history.rows = [{"job_id": "1", "nombre_trabajo": "grafeno"}]
        router.next = RoutedRequest(intent=Intent.QUERY_DB, params={"filtro_busqueda": "grafeno"})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="historial de grafeno")
        assert "grafeno" in reply.text
        assert history.last_filter.name_contains == "grafeno"

    def test_history_empty(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.QUERY_DB, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="historial")
        assert "No se encontraron" in reply.text

    def test_unknown_intent_returns_help(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.UNKNOWN, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="asdf")
        assert reply.text == HELP_TEXT


class TestStructureGeneration:
    def test_structure_uploads_to_requesters_own_account(self, env):
        service, router, factory, _, _, structures, _ = env
        router.next = RoutedRequest(
            intent=Intent.MODIFY_STRUCTURE,
            params={"formula": "Si", "destino_remoto": "/home/alice/calc"},
        )
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="generá Si")
        assert "Estructura generada" in reply.text
        assert len(factory.gateways["alice"].uploads) == 1
        assert "bob" not in factory.gateways


class TestJobTracking:
    """El cierre del loop: enviar un cálculo activa su seguimiento;
    cancelarlo lo desactiva para no duplicar el aviso del monitor."""

    def test_successful_submit_tracks_the_job(self, env):
        service, router, factory, _, _, _, tracker = env
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
        )
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        reply = service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        assert len(tracker.tracked) == 1
        tracked = tracker.tracked[0]
        assert tracked.job_id == "4242"
        assert tracked.owner_id == ALICE.telegram_user_id
        assert tracked.ssh_user == "alice"
        assert not tracked.notified
        assert "Seguimiento activado" in reply.text

    def test_submit_without_parsable_job_id_does_not_track(self, env):
        service, router, factory, *_ , tracker = env
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
        )
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        # Simulamos un sbatch exitoso pero con salida no parseable.
        factory.gateways["alice"].submit_job = lambda req: CommandResult(
            ok=True, stdout="algo raro sin el patrón esperado"
        )
        # Como el token ya se consumió arriba, preparamos uno nuevo:
        prep2 = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        service.confirm(prep2.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert tracker.tracked == []  # nada que rastrear sin job_id

    def test_cancel_marks_tracked_job_as_notified(self, env):
        service, router, factory, *_, tracker = env
        # Alice ya tenía un trabajo en seguimiento (por ejemplo, enviado antes).
        tracker.track(TrackedJob(
            job_id="777", owner_id=ALICE.telegram_user_id, chat_id=1,
            ssh_user="alice", job_name="grafeno",
        ))
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "777"})
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá el 777")
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert ("777", ALICE.telegram_user_id) in tracker.notified_calls
        assert tracker.active_jobs() == []  # ya no lo vuelve a consultar el monitor
