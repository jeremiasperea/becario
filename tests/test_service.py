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
    CalcDirResult,
    CalcKind,
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
    def __init__(self):
        self.next: RoutedRequest = RoutedRequest(intent=Intent.UNKNOWN, params={})

    def route(self, user_text: str) -> RoutedRequest:
        return self.next

    def extract_params(self, user_text: str) -> dict:
        return dict(self.next.params)


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
        self.uploads: list[tuple[str, str]] = []
        self.uploaded_dirs: list[tuple[str, str]] = []
        self.concatenated: list[tuple[tuple[str, ...], str]] = []
        # POTCARs "presentes" en la biblioteca del cluster de mentira:
        self.existing_files: set[str] = {"/potcars/Zr_sv/POTCAR", "/potcars/W/POTCAR"}
        # Archivos remotos legibles (para consultas de resultados):
        self.remote_files: dict[str, str] = {}

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

    def read_file(self, remote_path: str) -> Optional[str]:
        return self.remote_files.get(remote_path)

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

    def generate(self, req: VaspCalcRequest) -> CalcDirResult:
        self.requests.append(req)
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
        ["/tmp/../etc", "relativa/pruebas", "/tmp/x; rm -rf /", "/tmp/$(id)"],
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
