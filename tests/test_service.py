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
    _LISTING_MAX_CHARS,
    _VIEW_FILE_MAX_BYTES,
    _truncate_listing,
)
from becario.domain.models import (
    CalcDirResult,
    CalcKind,
    ClusterIdentity,
    CommandResult,
    HistoryFilter,
    Intent,
    JobId,
    JobStatus,
    PendingPlan,
    Plan,
    PlanStep,
    RoutedRequest,
    SlurmJobRequest,
    StructureRequest,
    StructureResult,
    TrackedJob,
    VaspCalcRequest,
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
    """`route` devuelve `Plan` (task 3.2/4.3). Los ~70 sitios de test
    existentes siguen fijando `router.next = RoutedRequest(...)`: `route`
    lo envuelve en un plan de un solo paso, así que el comportamiento de
    un solo paso queda byte-idéntico sin tocar esos call sites. Los tests
    de planes multi-paso fijan `router.next_plan` directamente."""

    def __init__(self):
        self.next: RoutedRequest = RoutedRequest(intent=Intent.UNKNOWN, params={})
        self.next_plan: Optional[Plan] = None
        # `extract_edit` (tarea 5.1): los tests de edición multi-paso fijan
        # esto directamente en vez de simular el LLM real.
        self.next_edit: tuple[Optional[int], dict] = (None, {})
        self.extract_edit_calls: list[tuple[str, str]] = []
        self.next_decomposition: list[str] = []
        self.decompose_calls: list[str] = []
        self.route_calls: list[str] = []
        self.routes_queue: list[Plan] = []

    def route(self, user_text: str) -> Plan:
        self.route_calls.append(user_text)
        if self.routes_queue:
            return self.routes_queue.pop(0)
        if self.next_plan is not None:
            return self.next_plan
        return Plan(steps=[PlanStep(action=self.next.intent, parametros=dict(self.next.params))])

    def extract_params(self, user_text: str) -> dict:
        return dict(self.next.params)

    def extract_edit(self, plan_context: str, user_text: str) -> tuple[Optional[int], dict]:
        self.extract_edit_calls.append((plan_context, user_text))
        return self.next_edit

    def decompose(self, user_text: str) -> list[str]:
        # Instrucciones fijadas por el test; al rutearse cada una, `route`
        # consume `routes_queue` si el test lo cargó.
        self.decompose_calls.append(user_text)
        return list(self.next_decomposition)


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
        self.made_dirs: list[str] = []
        # Si se define, make_directory devuelve esto (para simular fallos):
        self.make_directory_result: CommandResult | None = None
        self.listed_dirs: list[str] = []
        # Si se define, list_directory devuelve esto (fallos o listados largos):
        self.list_directory_result: CommandResult | None = None
        self.uploads: list[tuple[str, str]] = []
        self.uploaded_dirs: list[tuple[str, str]] = []
        self.concatenated: list[tuple[tuple[str, ...], str]] = []
        # POTCARs "presentes" en la biblioteca del cluster de mentira:
        self.existing_files: set[str] = {"/potcars/Zr_sv/POTCAR", "/potcars/W/POTCAR"}
        # Archivos remotos legibles (para consultas de resultados):
        self.remote_files: dict[str, str] = {}
        # Rutas pasadas a read_file (para verificar qué se intentó leer):
        self.read_calls: list[str] = []
        # `max_bytes` de cada read_file, en paralelo a read_calls:
        self.read_max_bytes: list = []

    def submit_job(self, req: SlurmJobRequest) -> CommandResult:
        self.submitted.append(req)
        return CommandResult(ok=True, stdout="Submitted batch job 4242", job_id="4242")

    def cancel_job(self, job_id: JobId) -> CommandResult:
        self.cancelled.append(job_id)
        return CommandResult(ok=True, stdout=f"Job {job_id} cancelado")

    def job_status(self, job_id: Optional[JobId]) -> CommandResult:
        self.status_calls.append(job_id)
        return CommandResult(ok=True, stdout="JOBID  NAME  STATE")

    def make_directory(self, path: str) -> CommandResult:
        self.made_dirs.append(path)
        if self.make_directory_result is not None:
            return self.make_directory_result
        return CommandResult(ok=True, stdout=f"Directorio listo: {path}")

    def list_directory(self, path: str) -> CommandResult:
        self.listed_dirs.append(path)
        if self.list_directory_result is not None:
            return self.list_directory_result
        return CommandResult(
            ok=True, stdout="total 8\ndrwxr-xr-x 2 alice alice 4,0K corrida_1"
        )

    def upload_file(self, local_path: str, remote_path: str) -> CommandResult:
        self.uploads.append((local_path, remote_path))
        return CommandResult(ok=True, stdout=f"Archivo subido a {remote_path}")

    def upload_dir(self, local_dir: str, remote_dir: str) -> CommandResult:
        self.uploaded_dirs.append((local_dir, remote_dir))
        return CommandResult(ok=True, stdout=f"Directorio subido a {remote_dir}")

    def home_dir(self) -> Optional[str]:
        return f"/home/{self.ssh_user}"

    def file_exists(self, remote_path: str) -> bool:
        return remote_path in self.existing_files

    def list_dir(self, remote_dir: str) -> Optional[list[str]]:
        names = [
            p[len(remote_dir) + 1:].split("/")[0]
            for p in self.remote_files if p.startswith(remote_dir + "/")
        ]
        return sorted(set(names)) or None

    def read_file(
        self, remote_path: str, max_bytes: Optional[int] = None
    ) -> Optional[str]:
        self.read_calls.append(remote_path)
        self.read_max_bytes.append(max_bytes)
        content = self.remote_files.get(remote_path)
        if content is not None and max_bytes is not None:
            # Simula el tope real: solo baja el prefijo pedido.
            content = content.encode()[:max_bytes].decode(errors="replace")
        return content

    def concat_files(self, sources: list[str], dest: str) -> CommandResult:
        self.concatenated.append((tuple(sources), dest))
        return CommandResult(ok=True)

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


class FakeCalcInputGenerator:
    def __init__(self):
        self.requests: list[VaspCalcRequest] = []
        self.atoms_seen: list = []

    def generate(self, req: VaspCalcRequest, atoms=None) -> CalcDirResult:
        self.requests.append(req)
        self.atoms_seen.append(atoms)
        encut_values = (
            req.scan_values() if req.calc_kind is CalcKind.ENCUT_SCAN else [req.encut]
        )
        return CalcDirResult(
            local_dir="/tmp/runs/fake_run",
            run_name=f"{req.formula}_{req.calc_kind.value}_x",
            files=["INCAR", "KPOINTS", "POSCAR", "run_vasp.sh"],
            elements=[req.formula],
            chemical_formula=req.formula,
            n_atoms=2,
            cell_summary="a=3.230 Å, b=3.230 Å, c=5.170 Å",
            kpoints=(9, 9, 6),
            calc_kind=req.calc_kind,
            encut_values=encut_values,
        )


class FakeCalcRuns:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, owner_id, job_id, job_name, fingerprint, run_dir) -> None:
        self.rows.append({
            "owner_id": owner_id, "job_id": job_id, "job_name": job_name,
            "fingerprint": fingerprint, "run_dir": run_dir,
            "fecha": "2026-07-11 10:00",
        })

    def find_by_name(self, owner_id, job_name, limit=3) -> list[dict]:
        found = [
            r for r in reversed(self.rows)
            if r["owner_id"] == owner_id and r["job_name"] == job_name
        ]
        return found[:limit]

    def find_recent(self, owner_id, job_name_prefix="", limit=5) -> list[dict]:
        found = [
            r for r in reversed(self.rows)
            if r["owner_id"] == owner_id
            and r["job_name"].startswith(job_name_prefix)
        ]
        return found[:limit]


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
        calc_inputs=FakeCalcInputGenerator(),
        potcar_dir="/potcars",
        remote_base="becario_runs",
        calc_runs=FakeCalcRuns(),
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


class TestPendingPlanShape:
    """El `ConfirmationStore` guarda `PendingPlan` (no `PendingAction`
    suelto): los tres sitios que arman una confirmación (`_prepare_submit`,
    `_prepare_cancel`, `_prepare_calc`) tienen que envolver su acción en un
    plan de un solo paso. Un plan de un paso se comporta byte-a-byte igual
    que antes (HC2), pero la FORMA que queda guardada en el store cambia."""

    def test_prepare_submit_stores_a_one_step_plan(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
        )
        token = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="x"
        ).confirmation_token

        pending = service._confirmations.peek(token)
        assert isinstance(pending, PendingPlan)
        assert pending.requester_id == ALICE.telegram_user_id
        assert len(pending.steps) == 1
        assert pending.steps[0].intent == Intent.SUBMIT_SLURM

    def test_prepare_cancel_stores_a_one_step_plan(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "777"})
        token = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá el 777"
        ).confirmation_token

        pending = service._confirmations.peek(token)
        assert isinstance(pending, PendingPlan)
        assert len(pending.steps) == 1
        assert pending.steps[0].intent == Intent.CANCEL_JOB

    def test_prepare_calc_stores_a_one_step_plan(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={"formula": "W", "tipo_calculo": "relajacion"},
        )
        token = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="relajá W"
        ).confirmation_token

        pending = service._confirmations.peek(token)
        assert isinstance(pending, PendingPlan)
        assert len(pending.steps) == 1
        assert pending.steps[0].intent == Intent.SUBMIT_SLURM  # PREPARE_CALC deja un SUBMIT pendiente


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


class TestCreateDirectory:
    """`crear_directorio` es no destructivo (mkdir -p): ejecuta directo,
    sin confirmación, pero la ruta pasa igual por el dominio."""

    def test_creates_directory_on_requesters_own_account(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.CREATE_DIR, params={"destino_remoto": "/home/alice/pruebas"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="creame la carpeta pruebas"
        )
        assert not reply.needs_confirmation
        assert "Directorio listo" in reply.text
        assert factory.gateways["alice"].made_dirs == ["/home/alice/pruebas"]
        assert "bob" not in factory.gateways

    @pytest.mark.parametrize(
        "evil",
        ["/tmp/../etc", "../fuera/del/workspace", "/tmp/x; rm -rf /", "/tmp/$(id)"],
    )
    def test_invalid_path_never_reaches_the_gateway(self, env, evil):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CREATE_DIR, params={"destino_remoto": evil})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="crea la carpeta")
        assert "Ruta inválida" in reply.text
        assert factory.gateways["alice"].made_dirs == []

    def test_missing_path_asks_for_it(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.CREATE_DIR, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="crea una carpeta")
        assert "ruta" in reply.text.lower()
        assert factory.gateways["alice"].made_dirs == []

    def test_gateway_failure_is_reported_to_the_user(self, env):
        service, router, factory, *_ = env
        # Pre-sembramos el gateway cacheado de alice con un mkdir que falla.
        gateway = FakeCluster("alice")
        gateway.make_directory_result = CommandResult(
            ok=False, stderr="mkdir: permiso denegado"
        )
        factory.gateways["alice"] = gateway
        router.next = RoutedRequest(
            intent=Intent.CREATE_DIR, params={"destino_remoto": "/home/alice/pruebas"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="creame la carpeta pruebas"
        )
        assert reply.text.startswith("❌")
        assert "permiso denegado" in reply.text
        assert gateway.made_dirs == ["/home/alice/pruebas"]


class TestListFiles:
    """`listar_archivos` es de solo lectura: ejecuta directo, sin
    confirmación, pero la ruta pasa igual por el dominio."""

    def test_lists_explicit_path_on_requesters_own_account(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.LIST_FILES, params={"destino_remoto": "/data/becario_runs"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="qué archivos hay"
        )
        assert not reply.needs_confirmation
        assert reply.monospace
        assert "corrida_1" in reply.text
        assert factory.gateways["alice"].listed_dirs == ["/data/becario_runs"]
        assert "bob" not in factory.gateways

    def test_missing_path_falls_back_to_remote_base(self, env):
        # remote_base es relativa ("becario_runs"): se resuelve contra el
        # home remoto, igual que al subir una corrida.
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.LIST_FILES, params={})
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="mostrame los archivos"
        )
        assert factory.gateways["alice"].listed_dirs == ["/home/alice/becario_runs"]
        assert "corrida_1" in reply.text

    @pytest.mark.parametrize(
        "evil",
        ["/tmp/../etc", "../fuera/del/workspace", "/tmp/x; rm -rf /", "/tmp/$(id)"],
    )
    def test_invalid_path_never_reaches_the_gateway(self, env, evil):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.LIST_FILES, params={"destino_remoto": evil})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="listá")
        assert "Ruta inválida" in reply.text
        assert factory.gateways["alice"].listed_dirs == []

    def test_gateway_failure_is_reported_to_the_user(self, env):
        service, router, factory, *_ = env
        gateway = FakeCluster("alice")
        gateway.list_directory_result = CommandResult(
            ok=False, stderr="ls: permiso denegado"
        )
        factory.gateways["alice"] = gateway
        router.next = RoutedRequest(
            intent=Intent.LIST_FILES, params={"destino_remoto": "/data/privado"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="listá /data/privado"
        )
        assert reply.text.startswith("❌")
        assert "permiso denegado" in reply.text

    def test_long_listing_is_truncated(self, env):
        service, router, factory, *_ = env
        gateway = FakeCluster("alice")
        long_listing = "\n".join(f"-rw-r--r-- 1 alice alice 1K archivo_{i}.dat" for i in range(200))
        assert len(long_listing) > 3500  # el fake tiene que superar el límite
        gateway.list_directory_result = CommandResult(ok=True, stdout=long_listing)
        factory.gateways["alice"] = gateway
        router.next = RoutedRequest(
            intent=Intent.LIST_FILES, params={"destino_remoto": "/data/becario_runs"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="listá las corridas"
        )
        assert len(reply.text) < 3700  # listado acotado + encabezado
        assert "(listado truncado)" in reply.text
        # Se corta en un borde de línea: la última entrada visible está entera.
        assert "archivo_0.dat" in reply.text

    def test_empty_directory_listing_passes_through(self, env):
        service, router, factory, *_ = env
        gateway = FakeCluster("alice")
        gateway.list_directory_result = CommandResult(ok=True, stdout="total 0")
        factory.gateways["alice"] = gateway
        router.next = RoutedRequest(
            intent=Intent.LIST_FILES, params={"destino_remoto": "/data/vacia"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="listá /data/vacia"
        )
        assert reply.text == "📂 /data/vacia:\ntotal 0"

    def test_unresolvable_home_asks_to_check_connection(self, env):
        # remote_base relativa y home remoto irresoluble: no hay ruta que
        # listar, se avisa en vez de mandar un ls sin sentido.
        service, router, factory, *_ = env
        gateway = FakeCluster("alice")
        gateway.home_dir = lambda: None
        factory.gateways["alice"] = gateway
        router.next = RoutedRequest(intent=Intent.LIST_FILES, params={})
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="mostrame los archivos"
        )
        assert "No pude resolver el home remoto" in reply.text
        assert gateway.listed_dirs == []


class TestViewFile:
    """`ver_archivo` muestra el contenido de UN archivo (solo lectura). Dos
    formas de identificarlo: nombre suelto resuelto contra la última corrida,
    o ruta absoluta directa. El nombre nunca puede escaparse del directorio."""

    RUN_DIR = "/data/runs/zr_relax"

    def _seed_run(self, service, factory, files=None):
        service._calc_runs.add(
            ALICE.telegram_user_id, "9", "Zr_relajacion", "{}", self.RUN_DIR
        )
        cluster = factory.for_identity(ALICE)
        for name, content in (files or {}).items():
            cluster.remote_files[f"{self.RUN_DIR}/{name}"] = content
        return cluster

    def _ask(self, service, router, params):
        router.next = RoutedRequest(intent=Intent.VIEW_FILE, params=params)
        return service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

    def test_by_name_resolves_against_latest_run(self, env):
        service, router, factory, *_ = env
        cluster = self._seed_run(service, factory, {"CONTCAR": "linea 1\nlinea 2\n"})
        reply = self._ask(service, router, {"nombre_archivo": "CONTCAR"})
        assert not reply.needs_confirmation
        assert reply.monospace
        assert "linea 1" in reply.text
        assert f"{self.RUN_DIR}/CONTCAR" in reply.text
        assert cluster.read_calls == [f"{self.RUN_DIR}/CONTCAR"]

    def test_by_absolute_path_reads_directly(self, env):
        service, router, factory, *_ = env
        cluster = factory.for_identity(ALICE)
        cluster.remote_files["/home/alice/run/OSZICAR"] = "E0 = -17.0\n"
        reply = self._ask(service, router, {"destino_remoto": "/home/alice/run/OSZICAR"})
        assert reply.monospace
        assert "E0 = -17.0" in reply.text
        assert cluster.read_calls == ["/home/alice/run/OSZICAR"]

    @pytest.mark.parametrize("evil", ["..", "../secreto", "a/b", "x;id", "/etc/passwd"])
    def test_unsafe_name_never_reaches_the_gateway(self, env, evil):
        # Un nombre con separadores o '..' se rechaza en el dominio: nunca se
        # construye una ruta que se escape del directorio de la corrida.
        service, router, factory, *_ = env
        cluster = self._seed_run(service, factory, {"CONTCAR": "x"})
        reply = self._ask(service, router, {"nombre_archivo": evil})
        assert "inválido" in reply.text
        assert cluster.read_calls == []

    def test_missing_both_asks_what_to_view(self, env):
        service, router, factory, *_ = env
        reply = self._ask(service, router, {})
        assert "qué archivo" in reply.text
        assert not reply.ok

    def test_by_name_without_runs_is_reported(self, env):
        service, router, factory, *_ = env
        reply = self._ask(service, router, {"nombre_archivo": "CONTCAR"})
        assert "No encontré corridas" in reply.text
        assert not reply.ok

    def test_unreadable_file_is_reported(self, env):
        service, router, factory, *_ = env
        self._seed_run(service, factory, files={})  # corrida sin el archivo
        reply = self._ask(service, router, {"nombre_archivo": "OUTCAR"})
        assert reply.text.startswith("❌")
        assert not reply.ok

    def test_other_users_runs_are_invisible(self, env):
        service, router, factory, *_ = env
        service._calc_runs.add(BOB.telegram_user_id, "9", "Zr_relajacion", "{}", self.RUN_DIR)
        reply = self._ask(service, router, {"nombre_archivo": "CONTCAR"})
        assert "No encontré corridas" in reply.text

    def test_long_content_is_truncated(self, env):
        service, router, factory, *_ = env
        long_body = "\n".join(f"linea_{i}" for i in range(2000))
        assert len(long_body) > 3500
        self._seed_run(service, factory, {"CONTCAR": long_body})
        reply = self._ask(service, router, {"nombre_archivo": "CONTCAR"})
        assert "(archivo truncado)" in reply.text
        assert "linea_0" in reply.text
        assert len(reply.text) < 3700

    def test_read_is_size_bounded(self, env):
        # El nombre lo elige el usuario: un archivo gigante (WAVECAR/CHGCAR)
        # NUNCA se baja entero. read_file recibe un tope de bytes y solo se
        # decodifica ese prefijo — la respuesta igual entra en Telegram.
        service, router, factory, *_ = env
        huge = "A" * (5 * 1024 * 1024)  # 5 MiB
        cluster = self._seed_run(service, factory, {"WAVECAR": huge})
        reply = self._ask(service, router, {"nombre_archivo": "WAVECAR"})
        assert cluster.read_max_bytes == [_VIEW_FILE_MAX_BYTES]
        assert "(archivo truncado)" in reply.text
        assert len(reply.text) < 3700

    def test_both_forms_given_is_reported(self, env):
        # El prompt del router puede extraer nombre Y ruta a la vez: se
        # rechaza con un mensaje claro, sin tocar el gateway.
        service, router, factory, *_ = env
        cluster = self._seed_run(service, factory, {"CONTCAR": "x"})
        reply = self._ask(
            service, router,
            {"nombre_archivo": "CONTCAR", "destino_remoto": "/home/alice/run/CONTCAR"},
        )
        assert "solo uno" in reply.text
        assert not reply.ok
        assert cluster.read_calls == []

    def test_empty_run_dir_is_reported(self, env):
        service, router, factory, *_ = env
        service._calc_runs.add(ALICE.telegram_user_id, "9", "Zr_relajacion", "{}", "")
        reply = self._ask(service, router, {"nombre_archivo": "CONTCAR"})
        assert "no tiene directorio" in reply.text
        assert not reply.ok

    def test_empty_file_is_reported(self, env):
        service, router, factory, *_ = env
        self._seed_run(service, factory, {"CONTCAR": ""})
        reply = self._ask(service, router, {"nombre_archivo": "CONTCAR"})
        assert "(archivo vacío)" in reply.text
        assert reply.ok

    def test_by_name_without_calc_runs_configured(self, env):
        # Sin historial de corridas, resolver por nombre es imposible: se pide
        # la ruta absoluta en vez de fallar.
        service, router, factory, *_ = env
        service._calc_runs = None
        reply = self._ask(service, router, {"nombre_archivo": "CONTCAR"})
        assert "ruta absoluta" in reply.text
        assert not reply.ok


class TestTruncateListing:
    """El recorte protege el límite de 4096 caracteres de Telegram."""

    def test_text_at_the_limit_is_untouched(self):
        text = "x" * _LISTING_MAX_CHARS
        assert _truncate_listing(text) == text

    def test_one_char_over_the_limit_truncates(self):
        text = "y" * (_LISTING_MAX_CHARS + 1)
        out = _truncate_listing(text)
        assert "(listado truncado)" in out
        assert len(out) < len(text) + 30

    def test_single_long_line_is_hard_cut(self):
        # Sin newline previo al límite no hay borde de línea: corte duro.
        text = "z" * (_LISTING_MAX_CHARS * 2)
        out = _truncate_listing(text)
        assert out.startswith("z" * 100)
        assert "(listado truncado)" in out
        assert len(out) <= _LISTING_MAX_CHARS + 30


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


class TestPrepareCalc:
    """Cálculo VASP completo: generar inputs, subir, armar POTCAR y
    dejar el sbatch esperando confirmación."""

    def test_full_flow_generates_uploads_and_asks_confirmation(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={"formula": "W", "tipo_calculo": "relajacion"},
        )
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="relajá W")
        assert reply.needs_confirmation
        cluster = factory.gateways["alice"]
        # Se subió el directorio a la base remota resuelta contra el home:
        assert cluster.uploaded_dirs == [
            ("/tmp/runs/fake_run", "/home/alice/becario_runs/W_relajacion_x")
        ]
        # El POTCAR se armó con la variante disponible (W, sin sufijo):
        assert cluster.concatenated == [
            (("/potcars/W/POTCAR",), "/home/alice/becario_runs/W_relajacion_x/POTCAR")
        ]
        # Pero nada se envió todavía:
        assert cluster.submitted == []

    def test_potcar_variant_lookup_prefers_sv(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC, params={"formula": "Zr", "red_cristalina": "hcp"}
        )
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        sources, _ = factory.gateways["alice"].concatenated[0]
        assert sources == ("/potcars/Zr_sv/POTCAR",)

    def test_missing_potcar_aborts_before_upload(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(intent=Intent.PREPARE_CALC, params={"formula": "Si"})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert not reply.needs_confirmation
        assert "POTCAR" in reply.text
        assert factory.gateways["alice"].uploaded_dirs == []

    def test_confirm_submits_and_tracks_workflow(self, env):
        service, router, factory, *_, tracker = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={
                "formula": "Zr",
                "tipo_calculo": "convergencia_encut",
                "encut_min": 250,
                "encut_max": 400,
                "encut_paso": 50,
            },
        )
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        cluster = factory.gateways["alice"]
        assert len(cluster.submitted) == 1
        submitted = cluster.submitted[0]
        assert submitted.script_path.endswith("/run_vasp.sh")
        assert tracker.tracked[0].workflow == "encut_scan"

    def test_encut_range_reaches_generator(self, env):
        service, router, *_ = env
        generator = service._calc_inputs
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={
                "formula": "Zr",
                "tipo_calculo": "convergencia_encut",
                "encut_min": 250,
                "encut_max": 400,
            },
        )
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert generator.requests[0].encut_values == [250, 300, 350, 400]

    def test_encut_range_without_tipo_infers_scan(self, env):
        """gemma3:4b a veces extrae el rango pero omite tipo_calculo."""
        service, router, *_ = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={"formula": "Zr", "encut_min": 250, "encut_max": 400},
        )
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        req = service._calc_inputs.requests[0]
        assert req.calc_kind is CalcKind.ENCUT_SCAN
        assert req.encut_values == [250, 300, 350, 400]

    def test_single_calc_does_not_tag_workflow(self, env):
        service, router, factory, *_, tracker = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={"formula": "W", "tipo_calculo": "relajacion"},
        )
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert tracker.tracked[0].workflow == ""

    def test_missing_formula_asks_for_it(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.PREPARE_CALC, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert not reply.needs_confirmation
        assert "material" in reply.text

    def test_unconfigured_potcar_dir_warns(self, env):
        service, router, *_ = env
        service._potcar_dir = ""
        router.next = RoutedRequest(intent=Intent.PREPARE_CALC, params={"formula": "W"})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert "BECARIO_POTCAR_DIR" in reply.text

    def test_evil_params_rejected(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.PREPARE_CALC,
            params={"formula": "W", "particion": "cpu; rm -rf /"},
        )
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert not reply.needs_confirmation
        assert factory.gateways["alice"].uploaded_dirs == []


class TestDuplicateDetection:
    """Aviso de "esto ya se corrió" (idéntico o muy similar), integrado a
    la confirmación: Confirmar = correr igual, Modificar = usar de base."""

    W_RELAX = {"formula": "W", "tipo_calculo": "relajacion"}

    def _prepare(self, service, router, params) -> "Reply":
        router.next = RoutedRequest(intent=Intent.PREPARE_CALC, params=dict(params))
        return service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

    def test_first_run_has_no_warning(self, env):
        service, router, *_ = env
        reply = self._prepare(service, router, self.W_RELAX)
        assert "🔁" not in reply.text
        assert reply.allow_modify

    def test_unconfirmed_run_is_not_recorded(self, env):
        service, router, *_ = env
        self._prepare(service, router, self.W_RELAX)  # queda sin confirmar
        reply = self._prepare(service, router, self.W_RELAX)
        assert "🔁" not in reply.text

    def test_identical_rerun_warns_exact(self, env):
        service, router, *_ = env
        prep = self._prepare(service, router, self.W_RELAX)
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        reply = self._prepare(service, router, self.W_RELAX)
        assert "ESTO YA SE CORRIÓ" in reply.text
        assert reply.text.startswith("🔁")  # el aviso va PRIMERO, no enterrado
        assert "4242" in reply.text  # el job de la corrida previa
        assert reply.needs_confirmation  # igual se puede correr de nuevo

    def test_similar_rerun_warns_similar(self, env):
        service, router, *_ = env
        prep = self._prepare(service, router, self.W_RELAX)
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        reply = self._prepare(service, router, {**self.W_RELAX, "encut": 600})
        assert "muy similar" in reply.text
        assert "ESTO YA SE CORRIÓ" not in reply.text

    def test_other_users_runs_do_not_warn(self, env):
        service, router, *_ = env
        prep = self._prepare(service, router, self.W_RELAX)
        service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        router.next = RoutedRequest(intent=Intent.PREPARE_CALC, params=dict(self.W_RELAX))
        reply = service.handle_text(chat_id=2, user_id=BOB.telegram_user_id, text="x")
        assert "🔁" not in reply.text


class TestPlanModification:
    """Botón ✏️ Modificar: el próximo mensaje describe el cambio y se
    mezcla con el plan original."""

    SCAN = {
        "formula": "Zr", "tipo_calculo": "convergencia_encut",
        "encut_min": 250, "encut_max": 400,
    }

    def _prepare(self, service, router, params) -> "Reply":
        router.next = RoutedRequest(intent=Intent.PREPARE_CALC, params=dict(params))
        return service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

    def test_modify_then_message_merges_params(self, env):
        service, router, *_ = env
        prep = self._prepare(service, router, self.SCAN)
        ask = service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert "qué querés cambiar" in ask.text

        # El próximo mensaje solo trae el cambio; el resto se mantiene.
        router.next = RoutedRequest(intent=Intent.PREPARE_CALC, params={"encut_max": 500})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="subilo a 500")
        assert reply.needs_confirmation
        req = service._calc_inputs.requests[-1]
        assert req.encut_values == [250, 300, 350, 400, 450, 500]
        assert req.formula == "Zr"

    def test_modify_consumes_the_pending_action(self, env):
        service, router, *_ = env
        prep = self._prepare(service, router, self.SCAN)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        reply = service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert "expiró" in reply.text

    def test_foreign_modification_is_blocked(self, env):
        service, router, *_ = env
        prep = self._prepare(service, router, self.SCAN)
        reply = service.start_modification(prep.confirmation_token, requester_id=BOB.telegram_user_id)
        assert "no te pertenece" in reply.text.lower()
        # La acción de Alice sigue viva:
        assert service.confirm(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

    def test_cancel_action_is_not_modifiable(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "777"})
        prep = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá el 777")
        assert not prep.allow_modify
        reply = service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert "no admite" in reply.text

    def test_unintelligible_change_keeps_the_plan_for_retry(self, env):
        """El plan NO se descarta solo: se reintenta hasta que salga o el
        usuario cancele explícitamente."""
        service, router, *_ = env
        prep = self._prepare(service, router, self.SCAN)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        router.next = RoutedRequest(intent=Intent.UNKNOWN, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="ehh")
        assert "No entendí" in reply.text
        assert "cancelar" in reply.text  # explica cómo salir

        # Segundo intento, ahora entendible: el plan seguía vivo.
        router.next = RoutedRequest(intent=Intent.UNKNOWN, params={"nodos": 2})
        reply2 = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="2 nodos")
        assert reply2.needs_confirmation
        assert service._calc_inputs.requests[-1].nodes == 2
        # Y conservó el resto del plan original:
        assert service._calc_inputs.requests[-1].encut_values == [250, 300, 350, 400]

    def test_explicit_cancel_discards_the_plan(self, env):
        service, router, *_ = env
        prep = self._prepare(service, router, self.SCAN)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cancelalo")
        assert "descartado" in reply.text
        # El modo edición terminó: el próximo mensaje rutea normal.
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={})
        reply2 = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="estado")
        assert "Estado" in reply2.text

    def test_expired_token_cannot_start_modification(self, env):
        service, *_ = env
        reply = service.start_modification("nope", requester_id=ALICE.telegram_user_id)
        assert "expiró" in reply.text


class TestMultiStepPlanModification:
    """Generalización de ✏️ Modificar a planes de VARIOS pasos (tarea 5.1,
    diseño §3.2/§4.6): el cambio se aplica al paso correcto (semántico o
    explícito «paso N»), un targeting ambiguo NUNCA fusiona ni ejecuta, y
    un cambio aceptado SIEMPRE re-arma el plan entero desde cero (nunca
    parchea en silencio un payload ya materializado)."""

    def _prepare_composite(self, service, router) -> "Reply":
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/home/alice/run"}),
            PlanStep(action=Intent.SUBMIT_SLURM, parametros={"script_remoto": "/home/alice/run/x.sh"}),
        ])
        return service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

    def test_semantic_targeting_updates_the_destructive_step_and_rematerializes_prefix(self, env):
        service, router, factory, *_ = env
        prep = self._prepare_composite(service, router)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert len(factory.gateways["alice"].made_dirs) == 1  # materializado al armar el plan

        # El LLM identifica semánticamente el paso destructivo (2), sin
        # que el usuario diga "paso 2".
        router.next_edit = (2, {"nodos": 4})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="usá 4 nodos")
        assert reply.needs_confirmation

        cluster = factory.gateways["alice"]
        assert len(cluster.made_dirs) == 2  # re-PREPARE del plan entero, no un parche puntual
        service.confirm(reply.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert cluster.submitted[-1].nodes == 4

    def test_explicit_step_targeting_edits_a_prefix_step(self, env):
        service, router, factory, *_ = env
        prep = self._prepare_composite(service, router)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        router.next_edit = (1, {"destino_remoto": "/home/alice/other"})
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="paso 1: usá /home/alice/other",
        )
        assert reply.needs_confirmation
        cluster = factory.gateways["alice"]
        assert cluster.made_dirs[-1] == "/home/alice/other"  # el paso 1 se re-materializó

    def test_ambiguous_edit_reprompts_without_changing_the_plan(self, env):
        service, router, factory, *_ = env
        prep = self._prepare_composite(service, router)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        router.next_edit = (None, {"nodos": 4})  # el LLM no está seguro de a qué paso
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cambialo")
        assert not reply.needs_confirmation
        assert "paso" in reply.text.lower()
        assert len(factory.gateways["alice"].made_dirs) == 1  # nada se re-materializó ni ejecutó

        # TTL refrescado: el plan sigue vivo, un segundo intento explícito
        # todavía puede aplicarse (no hizo falta reiniciar con ✏️).
        router.next_edit = (2, {"nodos": 4})
        reply2 = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="paso 2: 4 nodos")
        assert reply2.needs_confirmation

    def test_out_of_range_target_index_is_ambiguous(self, env):
        service, router, factory, *_ = env
        prep = self._prepare_composite(service, router)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        router.next_edit = (5, {"nodos": 4})  # el plan solo tiene 2 pasos
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cambialo")
        assert not reply.needs_confirmation
        assert len(factory.gateways["alice"].made_dirs) == 1

    def test_empty_delta_with_target_index_is_ambiguous(self, env):
        service, router, factory, *_ = env
        prep = self._prepare_composite(service, router)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        router.next_edit = (2, {})  # target sin cambio real: nada que aplicar
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="paso 2: algo")
        assert not reply.needs_confirmation
        assert len(factory.gateways["alice"].made_dirs) == 1

    def test_edit_never_patches_payload_silently_full_reprepare(self, env):
        """El cambio no se aplica "parchando" el `PendingAction.payload`
        del paso destructivo: se re-arma el plan entero y se re-valida
        contra el modelo de dominio (`SlurmJobRequest`), como una
        preparación nueva — un valor inválido se rechaza igual que en un
        pedido nuevo, no queda un payload corrupto a mitad de camino."""
        service, router, factory, *_ = env
        prep = self._prepare_composite(service, router)
        service.start_modification(prep.confirmation_token, requester_id=ALICE.telegram_user_id)

        router.next_edit = (2, {"nodos": "no-es-un-numero"})
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="paso 2: nodos raros",
        )
        assert not reply.needs_confirmation
        assert "inválidos" in reply.text.lower()
        assert factory.gateways["alice"].submitted == []  # nunca se envió nada


class TestCompositePlans:
    """`handle_text` construye un `Plan` vía el router. Un plan de un solo
    paso sigue el camino de hoy, byte a byte (HC1/HC2, cubierto por el
    resto de la suite vía `FakeRouter.route`). Estos tests cubren la
    composición de VARIOS pasos (SR2/SR5): materialización en orden en el
    momento de construir el plan, corte en el primer fallo, y confirmación
    reservada SOLO para la cola destructiva final."""

    def test_two_safe_steps_execute_in_order_with_no_confirmation(self, env):
        service, router, factory, *_ = env
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/home/alice/a"}),
            PlanStep(action=Intent.LIST_FILES, parametros={"destino_remoto": "/home/alice/b"}),
        ])
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

        assert not reply.needs_confirmation
        cluster = factory.gateways["alice"]
        assert cluster.made_dirs == ["/home/alice/a"]  # el paso 1 corrió
        assert cluster.listed_dirs == ["/home/alice/b"]  # el paso 2 también
        assert reply.text.splitlines()[0].startswith("1. ✅")
        assert reply.text.splitlines()[1].startswith("2. ✅")

    def test_middle_step_failure_reports_executed_failed_and_not_attempted(self, env):
        """SR5: el paso 1 se reporta ejecutado, el 2 fallado con su
        mensaje, el 3 no intentado — y el paso 1 NO se deshace."""
        service, router, factory, *_ = env
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/home/alice/ok"}),
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/tmp/../etc"}),
            PlanStep(action=Intent.LIST_FILES, parametros={"destino_remoto": "/home/alice/b"}),
        ])
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

        assert not reply.needs_confirmation
        lines = reply.text.splitlines()
        assert lines[0].startswith("1. ✅")
        assert lines[1].startswith("2. ❌")
        assert "inválid" in lines[1].lower()
        assert lines[2] == "3. ⏸ omitido"

        cluster = factory.gateways["alice"]
        assert cluster.made_dirs == ["/home/alice/ok"]  # el paso 1 quedó hecho, no se deshizo
        assert cluster.listed_dirs == []  # el paso 3 NUNCA se invocó

    def test_composite_with_destructive_tail_materializes_prefix_then_asks_confirmation(self, env):
        service, router, factory, *_ = env
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/home/alice/run"}),
            PlanStep(action=Intent.SUBMIT_SLURM, parametros={"script_remoto": "/home/alice/run/x.sh"}),
        ])
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

        cluster = factory.gateways["alice"]
        assert cluster.made_dirs == ["/home/alice/run"]  # materializado al construir el plan
        assert cluster.submitted == []  # la cola destructiva todavía NO se ejecutó
        assert reply.needs_confirmation
        assert reply.confirmation_token

        service.confirm(reply.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert len(cluster.submitted) == 1  # recién ahora se ejecuta, una sola vez

    def test_composite_confirmation_shows_the_full_destructive_detail(self, env):
        # ADR-0003: se confirma exactamente lo que se va a ejecutar — el
        # detalle del sbatch (script, partición, nodos, tiempo) tiene que
        # estar visible en el texto, no solo el rótulo del paso.
        service, router, factory, *_ = env
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/home/alice/run"}),
            PlanStep(action=Intent.SUBMIT_SLURM, parametros={"script_remoto": "/home/alice/run/x.sh"}),
        ])
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")
        assert reply.needs_confirmation
        assert "script: /home/alice/run/x.sh" in reply.text
        assert "partición:" in reply.text
        assert "nodos:" in reply.text
        assert "tiempo:" in reply.text

    def test_composite_prefix_failure_blocks_the_destructive_tail(self, env):
        service, router, factory, *_ = env
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/tmp/../etc"}),
            PlanStep(action=Intent.SUBMIT_SLURM, parametros={"script_remoto": "/home/alice/run/x.sh"}),
        ])
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

        assert not reply.needs_confirmation
        cluster = factory.gateways["alice"]
        assert cluster.submitted == []  # la cola destructiva nunca se alcanzó
        assert reply.text.splitlines()[-1] == "2. ⏸ omitido"

    def test_unsupported_composite_step_is_rejected_fail_closed(self, env):
        """Un paso sin handler combinable (p. ej. `error`) rechaza el plan
        entero fail-closed. (`preparar_calculo` como cola SÍ se soporta:
        ver TestCalcTailIndividualConfirmations.)"""
        service, router, factory, *_ = env
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/home/alice/run"}),
            PlanStep(action=Intent.UNKNOWN, parametros={}),
        ])
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

        assert reply.text == HELP_TEXT
        assert not reply.needs_confirmation
        assert factory.gateways["alice"].made_dirs == []  # nada se ejecutó


_CONTCAR_ZR = (
    "Zr relajado\n"
    "1.0\n"
    "3.2300000 0.0000000 0.0000000\n"
    "-1.6150000 2.7972620 0.0000000\n"
    "0.0000000 0.0000000 5.1453900\n"
    "Zr\n"
    "2\n"
    "Direct\n"
    "0 0 0\n"
    "0.333 0.667 0.5\n"
)


class TestQueryResults:
    """'dame los parámetros de red del Zr' lee la celda de la corrida
    previa (CONTCAR si hubo relajación) en vez de contestar el historial."""

    RUN_DIR = "/data/runs/zr_relax"

    def _setup_run(self, service, factory, job_name="Zr_relajacion", with_contcar=True):
        service._calc_runs.add(
            ALICE.telegram_user_id, "9", job_name, "{}", self.RUN_DIR
        )
        cluster = factory.for_identity(ALICE)
        if with_contcar:
            cluster.remote_files[f"{self.RUN_DIR}/CONTCAR"] = _CONTCAR_ZR
            cluster.remote_files[f"{self.RUN_DIR}/OSZICAR"] = (
                "   1 F= -.17046660E+02 E0= -.17046660E+02  d E =0.0\n"
            )
        return cluster

    def _ask(self, service, router, params=None):
        router.next = RoutedRequest(
            intent=Intent.QUERY_RESULTS, params=params or {"formula": "Zr"}
        )
        return service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

    def test_lattice_parameters_from_contcar(self, env):
        service, router, factory, *_ = env
        self._setup_run(service, factory)
        reply = self._ask(service, router)
        assert "a = 3.2300" in reply.text
        assert "c = 5.1454" in reply.text
        assert "γ = 120.00°" in reply.text
        assert "E0 = -17.046660 eV" in reply.text
        assert "CONTCAR" in reply.text

    def test_prefers_relaxation_over_more_recent_scan(self, env):
        service, router, factory, *_ = env
        self._setup_run(service, factory)  # relajación (más vieja)
        service._calc_runs.add(  # barrido más reciente del mismo material
            ALICE.telegram_user_id, "10", "Zr_convergencia_encut", "{}", "/data/runs/zr_scan"
        )
        reply = self._ask(service, router)
        assert "Zr_relajacion" in reply.text

    def test_falls_back_to_poscar_when_no_contcar(self, env):
        service, router, factory, *_ = env
        cluster = self._setup_run(service, factory, with_contcar=False)
        cluster.remote_files[f"{self.RUN_DIR}/POSCAR"] = _CONTCAR_ZR
        reply = self._ask(service, router)
        assert "sin relajar" in reply.text
        assert "a = 3.2300" in reply.text

    def test_no_previous_runs(self, env):
        service, router, *_ = env
        reply = self._ask(service, router)
        assert "No encontré corridas" in reply.text

    def test_unreadable_cell_warns_with_run_dir(self, env):
        service, router, factory, *_ = env
        self._setup_run(service, factory, with_contcar=False)  # sin archivos
        reply = self._ask(service, router)
        assert "No pude leer la celda" in reply.text
        assert self.RUN_DIR in reply.text

    def test_other_users_runs_are_invisible(self, env):
        service, router, factory, *_ = env
        service._calc_runs.add(BOB.telegram_user_id, "9", "Zr_relajacion", "{}", self.RUN_DIR)
        reply = self._ask(service, router)
        assert "No encontré corridas" in reply.text


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


# ---------------------------------------------------------------------------
# Registro de decisiones del router (RouterDecisionLog)
# ---------------------------------------------------------------------------


class FakeDecisionLog:
    """Implementa RouterDecisionLog en memoria; registra llamadas."""

    def __init__(self):
        self.decisions: list[dict] = []
        self.outcomes: dict[int, str] = {}

    def add(self, chat_id, user_id, text, steps_json, latency_seconds) -> int:
        self.decisions.append({
            "chat_id": chat_id, "user_id": user_id, "text": text,
            "steps_json": steps_json, "latency_seconds": latency_seconds,
        })
        return len(self.decisions)  # ids 1..N

    def set_outcome(self, decision_id: int, outcome: str) -> None:
        self.outcomes[decision_id] = outcome


@pytest.fixture()
def env_with_log():
    router = FakeRouter()
    registry = FakeUserRegistry([ALICE, BOB])
    cluster_factory = FakeClusterGatewayFactory()
    decision_log = FakeDecisionLog()
    service = BecarioService(
        router=router,
        registry=registry,
        cluster_factory=cluster_factory,
        history=FakeHistory(),
        confirmations=InMemoryConfirmationStore(ttl_seconds=600),
        structures=FakeStructureBuilder(),
        job_tracker=FakeJobTracker(),
        calc_inputs=FakeCalcInputGenerator(),
        potcar_dir="/potcars",
        remote_base="becario_runs",
        calc_runs=FakeCalcRuns(),
        decision_log=decision_log,
    )
    return service, router, decision_log


class TestRouterDecisionLogging:
    def test_every_routed_message_is_logged(self, env_with_log):
        service, router, log = env_with_log
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={})
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="estado del cluster")
        assert len(log.decisions) == 1
        d = log.decisions[0]
        assert d["text"] == "estado del cluster"
        assert '"revisar_estado"' in d["steps_json"]
        assert d["latency_seconds"] >= 0

    def test_confirm_labels_decision_as_confirmed(self, env_with_log):
        service, router, log = env_with_log
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
        )
        token = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="corré el cálculo"
        ).confirmation_token
        service.confirm(token, requester_id=ALICE.telegram_user_id)
        assert log.outcomes == {1: "confirmed"}

    def test_reject_labels_decision_as_cancelled(self, env_with_log):
        service, router, log = env_with_log
        router.next = RoutedRequest(intent=Intent.CANCEL_JOB, params={"job_id": "777"})
        token = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="cancelá el 777"
        ).confirmation_token
        service.reject(token, requester_id=ALICE.telegram_user_id)
        assert log.outcomes == {1: "cancelled"}

    def test_failed_step_labels_decision_as_error(self, env_with_log):
        service, router, log = env_with_log
        # CREATE_DIR sin ruta: el handler valida y devuelve ok=False.
        router.next = RoutedRequest(intent=Intent.CREATE_DIR, params={})
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="creame la carpeta")
        assert log.outcomes == {1: "error"}

    def test_unresolved_decision_stays_unlabeled(self, env_with_log):
        service, router, log = env_with_log
        router.next = RoutedRequest(
            intent=Intent.SUBMIT_SLURM, params={"script_remoto": "/opt/calc.sh"}
        )
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="corré x")
        # Pendiente de confirmación sin respuesta humana: sin outcome.
        assert log.outcomes == {}

    def test_service_without_log_still_works(self, env):
        service, router, *_ = env
        router.next = RoutedRequest(intent=Intent.CHECK_STATUS, params={})
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="estado")
        assert reply.text  # no explota sin decision_log


class TestWorkspaceRelativePaths:
    """Las rutas relativas se anclan en la base de corridas (workspace).

    Motivo: el dominio exigía rutas absolutas y el usuario habla en
    relativo ('dentro de Zr…'), así que el LLM quedaba forzado a inventar
    prefijos absolutos — carpetas reales terminaron en /Zr y
    /home/becario_runs, invisibles para el listado."""

    def test_create_dir_relative_resolves_against_runs_base(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.CREATE_DIR, params={"destino_remoto": "Zr/bcc"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="dentro de Zr creá bcc"
        )
        assert factory.gateways["alice"].made_dirs == [
            "/home/alice/becario_runs/Zr/bcc"
        ]
        assert reply.ok

    def test_create_dir_absolute_passes_through(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.CREATE_DIR, params={"destino_remoto": "/data/proyectos/x"}
        )
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="creala")
        assert factory.gateways["alice"].made_dirs == ["/data/proyectos/x"]

    def test_list_files_relative_resolves_against_runs_base(self, env):
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.LIST_FILES, params={"destino_remoto": "Zr"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="mostrame Zr"
        )
        assert factory.gateways["alice"].listed_dirs == [
            "/home/alice/becario_runs/Zr"
        ]
        assert "corrida_1" in reply.text

    def test_relative_traversal_is_still_rejected(self, env):
        # La resolución NO relaja la seguridad: el dominio valida la ruta
        # final y un '..' la mata aunque venga disfrazado de relativa.
        service, router, factory, *_ = env
        router.next = RoutedRequest(
            intent=Intent.CREATE_DIR, params={"destino_remoto": "Zr/../../etc"}
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="creala"
        )
        assert "Ruta inválida" in reply.text
        assert factory.gateways["alice"].made_dirs == []


class TestCalcTailIndividualConfirmations:
    """Un plan que termina en UN solo `preparar_calculo` materializa el
    prefijo y emite su confirmación individual (camino de ADR-0006). Con
    VARIOS cálculos ya no hay N followups: es un batch (ADR-0007), ver
    `TestBatchConfirmation`."""

    def _plan(self, *steps):
        return Plan(steps=[PlanStep(action=a, parametros=p) for a, p in steps])

    def test_failed_prefix_omits_calcs(self, env):
        service, router, factory, *_ = env
        gateway = factory.gateways.setdefault("alice", FakeCluster("alice"))
        gateway.make_directory_result = CommandResult(ok=False, stderr="disco lleno")
        router.next_plan = self._plan(
            (Intent.CREATE_DIR, {"destino_remoto": "W"}),
            (Intent.PREPARE_CALC, {"formula": "Zr", "red_cristalina": "hcp"}),
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="creá W y corré Zr"
        )
        assert reply.followups == ()
        assert "omitido" in reply.text

    def test_calc_in_middle_is_still_rejected(self, env):
        # `preparar_calculo` seguido de otro paso NO es cola: fail-closed.
        service, router, factory, *_ = env
        router.next_plan = self._plan(
            (Intent.PREPARE_CALC, {"formula": "Zr", "red_cristalina": "hcp"}),
            (Intent.CREATE_DIR, {"destino_remoto": "W"}),
        )
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="raro"
        )
        assert reply.followups == ()
        assert factory.gateways["alice"].made_dirs == []


class TestDecomposition:
    """Un plan multi-paso con cálculos sin `formula` (la firma del techo
    de extracción del modelo en mensajes largos) se reintenta vía
    descomposición: instrucciones simples auto-contenidas, cada una
    ruteada por el camino corto confiable."""

    def _incomplete_plan(self):
        return Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "ZrO2"}),
            PlanStep(action=Intent.PREPARE_CALC, parametros={"tipo_calculo": "relajacion"}),
            PlanStep(action=Intent.PREPARE_CALC, parametros={"tipo_calculo": "relajacion"}),
        ])

    def test_incomplete_multicalc_plan_is_decomposed_then_batched(self, env):
        # El plan incompleto se descompone y recompone con material; al ser
        # multi-cálculo, desemboca en UNA confirmación de batch (ADR-0007) —
        # nada se materializa hasta confirmar.
        service, router, factory, *_ = env
        router.next_plan = self._incomplete_plan()
        router.next_decomposition = [
            "creá la carpeta ZrO2",
            "relajá el bulk de Zr bcc",
            "relajá el bulk de Zr fcc",
        ]
        router.routes_queue = [
            Plan(steps=[PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "ZrO2"})]),
            Plan(steps=[PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "bcc"})]),
            Plan(steps=[PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "fcc"})]),
        ]
        # El primer route() del handle_text consume la cola: recargamos con
        # el plan incompleto al frente.
        router.routes_queue.insert(0, self._incomplete_plan())
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="pedido largo de ZrO2"
        )
        assert router.decompose_calls == ["pedido largo de ZrO2"]
        assert factory.gateways["alice"].made_dirs == []  # nada se ejecutó
        assert reply.followups == ()
        assert reply.needs_confirmation and reply.confirmation_token
        assert "Batch" in reply.text

    def test_failed_decomposition_of_incomplete_plan_is_fail_closed(self, env):
        # No se pudo descomponer y el plan original es multi-cálculo sin
        # material: al ir a batch, el cálculo inválido lo aborta SIN efectos
        # (mejor que materializar a medias).
        service, router, factory, *_ = env
        router.next_plan = self._incomplete_plan()
        router.next_decomposition = []  # el descompositor no pudo
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="pedido largo"
        )
        assert factory.gateways["alice"].made_dirs == []  # nada se ejecutó
        assert reply.followups == ()
        assert not reply.needs_confirmation
        assert "material" in reply.text  # repregunta la fórmula

    def test_complete_plans_never_trigger_decomposition(self, env):
        service, router, factory, *_ = env
        router.next_plan = Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "W"}),
            PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "W", "red_cristalina": "bcc"}),
        ])
        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="ok")
        assert router.decompose_calls == []

    def test_compound_message_decomposes_without_routing_full_text(self, env):
        # Un mensaje largo/compuesto NUNCA debe pasar por el route() de
        # schema grande (hace timeout en CPU): va directo a decompose.
        service, router, factory, *_ = env
        compound = (
            "creá la carpeta ZrO2 y relajá el bulk de ZrO2 en las redes "
            "cristalinas rocksalt, zincblende y fluorite, cada una en su carpeta"
        )
        router.next_decomposition = [
            "creá la carpeta ZrO2",
            "relajá el bulk de Zr bcc",
            "relajá el bulk de Zr fcc",
        ]
        router.routes_queue = [
            Plan(steps=[PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "ZrO2"})]),
            Plan(steps=[PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "bcc"})]),
            Plan(steps=[PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "fcc"})]),
        ]
        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text=compound
        )
        assert router.decompose_calls == [compound]
        # El mensaje completo NO se ruteó: solo las instrucciones simples.
        assert compound not in router.route_calls
        assert router.route_calls == router.next_decomposition
        # Multi-cálculo -> batch: nada se materializa hasta confirmar.
        assert factory.gateways["alice"].made_dirs == []
        assert reply.followups == ()
        assert reply.needs_confirmation and reply.confirmation_token

    def test_compound_but_undecomposable_never_routes_full_text(self, env):
        # Parece compuesto pero decompose no pudo: NO se rutea el texto
        # entero (evita el timeout del schema grande); responde con la ayuda.
        service, router, factory, *_ = env
        compound = "relajá el bulk de W y también minimizá sus parámetros de red por favor"
        router.next_decomposition = []
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text=compound)
        assert router.decompose_calls == [compound]
        assert compound not in router.route_calls  # jamás el route directo
        assert reply.text == HELP_TEXT

    def test_three_lattice_compound_becomes_batch(self, env):
        # El pedido de 3 redes recompone a 7 pasos: entra en el cap de 11 y,
        # al ser multi-cálculo, sale como UNA confirmación de batch (ADR-0007)
        # sin routear el mega-mensaje ni materializar nada.
        service, router, factory, *_ = env
        compound = (
            "creá la carpeta ZrO2 y relajá el bulk de ZrO2 en las redes "
            "cristalinas rocksalt, zincblende y fluorite, cada una en su carpeta"
        )
        router.next_decomposition = [
            "creá la carpeta ZrO2",
            "relajá el bulk de Zr rocksalt en ZrO2/rocksalt",
            "relajá el bulk de Zr zincblende en ZrO2/zincblende",
            "relajá el bulk de Zr fluorite en ZrO2/fluorite",
        ]
        # Cada instrucción de cálculo rutea a [calc, crear_directorio]: 1 + 3*2 = 7.
        router.routes_queue = [
            Plan(steps=[PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "ZrO2"})]),
            Plan(steps=[
                PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "rocksalt"}),
                PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "ZrO2/rocksalt"}),
            ]),
            Plan(steps=[
                PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "zincblende"}),
                PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "ZrO2/zincblende"}),
            ]),
            Plan(steps=[
                PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "fluorite"}),
                PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "ZrO2/fluorite"}),
            ]),
        ]
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text=compound)
        assert compound not in router.route_calls
        assert factory.gateways["alice"].made_dirs == []  # nada se ejecutó
        assert reply.followups == ()
        assert reply.needs_confirmation and reply.confirmation_token
        assert "Batch" in reply.text


class TestBatchConfirmation:
    """Batch (ADR-0007): un plan multi-cálculo se confirma ENTERO. Nada toca
    el cluster hasta confirmar; al confirmar se ejecuta todo en orden con
    corte al primer fallo (misma semántica sin-rollback de ADR-0006)."""

    def _batch_plan(self):
        return Plan(steps=[
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "runs"}),
            PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "Zr", "red_cristalina": "hcp"}),
            PlanStep(action=Intent.PREPARE_CALC, parametros={"formula": "W", "red_cristalina": "bcc"}),
        ])

    def test_nothing_runs_until_confirmed_then_all_in_order(self, env):
        service, router, factory, *_ = env
        router.next_plan = self._batch_plan()
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="batch")
        gw = factory.gateways["alice"]
        # Preview: nada tocó el cluster.
        assert gw.made_dirs == [] and gw.submitted == [] and gw.uploaded_dirs == []
        assert reply.needs_confirmation and reply.confirmation_token
        # Confirmar ejecuta todo: crea la carpeta y encola los 2 trabajos.
        done = service.confirm(reply.confirmation_token, requester_id=ALICE.telegram_user_id)
        assert gw.made_dirs == ["/home/alice/becario_runs/runs"]
        assert len(gw.submitted) == 2
        assert done.ok

    def test_stops_on_first_failure(self, env):
        service, router, factory, *_ = env
        gw = factory.gateways.setdefault("alice", FakeCluster("alice"))
        gw.make_directory_result = CommandResult(ok=False, stderr="disco lleno")
        router.next_plan = self._batch_plan()
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="batch")
        done = service.confirm(reply.confirmation_token, requester_id=ALICE.telegram_user_id)
        # El mkdir falló: los cálculos quedan omitidos, sin envíos.
        assert gw.submitted == []
        assert "omitido" in done.text and not done.ok

    def test_reject_executes_nothing(self, env):
        service, router, factory, *_ = env
        router.next_plan = self._batch_plan()
        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="batch")
        service.reject(reply.confirmation_token, requester_id=ALICE.telegram_user_id)
        gw = factory.gateways["alice"]
        assert gw.made_dirs == [] and gw.submitted == []


class TestCompoundHeuristic:
    """`_looks_compound` decide a quién preguntar primero, no clasifica.
    Conservadora donde importa: un falso positivo es inocuo (decompose
    devuelve una instrucción y se rutea igual)."""

    @pytest.mark.parametrize("text", [
        "creá la carpeta ZrO2 y relajá el bulk de ZrO2 en las redes "
        "cristalinas rocksalt, zincblende y fluorite, cada una en su carpeta",
        "creá la carpeta W y relajá el bulk de W en las redes bcc, fcc y hcp",
        "generá el POSCAR de Si diamond y después creá la carpeta runs",
    ])
    def test_compound_messages_are_flagged(self, text):
        assert BecarioService._looks_compound(text) is True

    @pytest.mark.parametrize("text", [
        "relajá el bulk de W con red cristalina bcc",
        "curva de convergencia de ENCUT para Zr hcp de 250 a 450",
        "listá mi home",
        "mostrame el CONTCAR",
    ])
    def test_simple_messages_are_not_flagged(self, text):
        assert BecarioService._looks_compound(text) is False
