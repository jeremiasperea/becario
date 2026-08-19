"""Tests de infraestructura: SQLite real (en tmp), store con TTL, y
verificación de que los comandos SSH generados están correctamente quoteados.
"""
import json
import time

import pytest

from becario.domain.models import (
    ClusterIdentity,
    CommandFailureReason,
    CommandResult,
    HistoryFilter,
    Intent,
    JobId,
    JobStatus,
    PendingAction,
    PendingEdit,
    PendingPlan,
    SlurmJobRequest,
    TrackedJob,
)
from becario.infrastructure.ssh_gateway import (
    SSHClusterGateway,
    SSHClusterGatewayFactory,
)
from becario.infrastructure.storage import (
    InMemoryConfirmationStore,
    SQLiteChatLogRepository,
    SQLiteConfirmationStore,
    SQLiteHistoryRepository,
    SQLiteJobTracker,
    SQLitePendingEditStore,
    SQLiteRouterDecisionLog,
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


def _plan(requester_id: int = 1, created_at: float | None = None) -> PendingPlan:
    """`created_at` explícito para los tests de vencimiento.

    Sin esto, un test que necesita que algo venza DESPUES de otra cosa
    termina corriéndole una carrera al reloj; ver la nota de
    `test_purge_borra_los_vencidos_y_deja_las_lapidas`.
    """
    plan = PendingPlan(chat_id=1, requester_id=requester_id,
                       steps=[_action(requester_id)])
    if created_at is not None:
        plan.created_at = created_at
    return plan


def _nacido_vencido(requester_id: int = 1) -> PendingPlan:
    """Un plan que ya nace con el TTL cumplido.

    Reemplaza al par `ttl_seconds=0.01` + `time.sleep(0.05)` que tenían
    todos los tests de vencimiento de este módulo: el TTL se decide en el
    `created_at`, no en el reloj de pared, así que el test no puede
    perderle la carrera a su propio setup (el modo de fallo está contado en
    `test_purge_borra_los_vencidos_y_deja_las_lapidas`) y no cuesta 40 ms.

    Sirve igual para las dos implementaciones: `SQLiteConfirmationStore`
    persiste el `created_at` del plan tal cual —viaja dentro del `plan_json`
    y también en su propia columna— y recalcula el vencimiento al leerlo,
    así que la costura es la misma en memoria y en disco.
    """
    return _plan(requester_id, created_at=time.time() - 3600.0)


def _en_memoria(tmp_path, ttl: float) -> InMemoryConfirmationStore:
    return InMemoryConfirmationStore(ttl_seconds=ttl)


def _en_disco(tmp_path, ttl: float) -> SQLiteConfirmationStore:
    return SQLiteConfirmationStore(str(tmp_path / "becario.db"), ttl_seconds=ttl)


@pytest.mark.parametrize("crear_store", [_en_memoria, _en_disco],
                         ids=["en_memoria", "en_disco"])
class TestContratoConfirmationStore:
    """El contrato del puerto `ConfirmationStore`, contra sus DOS
    implementaciones a la vez.

    Antes estaba escrito tres veces —dos clases sobre el store en memoria,
    una sobre el de disco, con los mismos escenarios repetidos— y esa forma
    no verifica lo único que acá importa: que las dos contesten igual.

    Por qué importa: `main.py` cablea el de SQLite, así que
    `InMemoryConfirmationStore` ya no lo usa nadie en producción. Sobrevive
    como DOBLE de media suite (`test_service`, `test_tolerancia_router`,
    `test_tolerancia_incidentes`), y un doble que se parece lo suficiente
    para pasar y no lo suficiente para servir es una familia de bugs que ya
    mordió tres veces a este proyecto (ver `test_contrato_de_dobles.py`).
    Probarlo contra el mismo contrato que el real es lo que le da valor: si
    divergen, la divergencia se ve acá y no en producción.

    La unidad del contrato es `PendingPlan`, que es lo que declara el
    puerto. La vieja variante sobre `PendingAction` no sobrevive a la
    parametrización, y tampoco debería: el store en memoria es duck-typed
    (solo mira `.token`/`.expired()`), pero el de disco serializa `steps`,
    así que `PendingPlan` es lo único que las dos pueden prometer.
    """

    def test_put_pop_round_trip(self, crear_store, tmp_path):
        """Ida y vuelta, y la lápida del segundo toque.

        El `status` final no es decorativo: es la diferencia entre decirle a
        quien apretó ✅ dos veces «ya se hizo» y mandarlo a repetir un
        sbatch.
        """
        store = crear_store(tmp_path, 600.0)
        token = store.put(_plan())

        recuperado = store.pop(token)

        assert recuperado is not None
        assert recuperado.steps[0].payload == {"job_id": "1"}
        assert recuperado.steps[0].intent is Intent.CANCEL_JOB
        assert store.pop(token) is None  # segundo pop: ya consumido
        assert store.status(token) == "consumido"

    def test_peek_no_consume(self, crear_store, tmp_path):
        """`peek` existe para validar QUIÉN puede confirmar antes de
        descartar el plan pendiente: si consumiera, la validación se
        comería la acción."""
        store = crear_store(tmp_path, 600.0)
        token = store.put(_plan())

        assert store.peek(token) is not None
        assert store.peek(token) is not None  # sigue ahí
        assert store.status(token) == "vigente"
        assert store.pop(token) is not None
        assert store.peek(token) is None  # ahora sí, consumido

    def test_token_nunca_visto_es_desconocido(self, crear_store, tmp_path):
        """La cuarta respuesta de `status`, la que justifica que sean cuatro
        y no tres: de un token que nunca vimos NO se puede afirmar que se
        usó."""
        assert crear_store(tmp_path, 600.0).status("no-existe") == "desconocido"

    def test_pop_de_un_vencido_no_lo_marca_consumido(self, crear_store, tmp_path):
        """El TTL corta `peek`, `pop` y `status`; y el `pop` de un vencido
        NO deja lápida.

        Un plan vencido no se ejecutó. Si el pop lo marcara consumido, el
        siguiente `status` diría "consumido" y el usuario leería «ya se usó
        — la acción se hizo con el primer toque» sobre algo que nunca pasó.
        "vencido" lo manda a rehacer el pedido, que es lo correcto.
        """
        store = crear_store(tmp_path, 60.0)
        token = store.put(_nacido_vencido())

        assert store.peek(token) is None
        assert store.pop(token) is None
        assert store.status(token) == "vencido"

    def test_purge_borra_los_vencidos_y_deja_las_lapidas(self, crear_store, tmp_path):
        """TTL holgado y vencidos que nacen vencidos, en vez de TTL de 10 ms
        más un sleep.

        La versión anterior era intermitente y el modo de fallo enseña algo:
        entre el `put` y el `pop` del consumido hay dos transacciones SQLite
        contra un archivo real, y con la máquina cargada eso pasaba los
        10 ms de TTL. El `pop` encontraba el plan YA vencido y devolvía None
        SIN dejar lápida —a propósito, ver el docstring de `pop`—, así que
        esa fila seguía contando como vencida-sin-consumir: la purga daba 3
        y el `status` decía "vencido". El test le corría una carrera a su
        propio setup.
        """
        store = crear_store(tmp_path, 60.0)
        consumido = store.put(_plan())
        store.pop(consumido)
        store.put(_nacido_vencido())
        store.put(_nacido_vencido())

        assert store.purge_expired() == 2  # solo los vencidos sin consumir
        assert store.status(consumido) == "consumido"


class TestSQLiteConfirmationStore:
    """Lo que es propio del store en disco y no puede ir al contrato
    compartido.

    La diferencia que importa contra el store en memoria no es de API sino
    de vida útil: acá el plan sobrevive al proceso. Era lo único volátil
    del sistema y lo que hacía que un reinicio se comiera la conversación
    en silencio. Todo lo que sí comparte con el de memoria vive en
    `TestContratoConfirmationStore`.
    """

    def _store(self, tmp_path, ttl=600.0) -> SQLiteConfirmationStore:
        return SQLiteConfirmationStore(str(tmp_path / "becario.db"), ttl_seconds=ttl)

    def test_conserva_el_pedido_original_para_modificar(self, tmp_path):
        # `request_intent`/`request_params` son lo que hace editable un plan
        # (`allow_modify`); si no sobreviven al round-trip, el botón ✏️
        # desaparece después de un reinicio. En memoria esto no se puede
        # romper —el objeto guardado es el mismo—; acá pasa por serialización.
        store = self._store(tmp_path)
        accion = PendingAction(
            chat_id=1, requester_id=1, intent=Intent.SUBMIT_SLURM,
            description="d", payload={"a": 1},
            request_intent=Intent.PREPARE_CALC,
            request_params={"formula": "Zr", "red_cristalina": "hcp"},
        )
        plan = PendingPlan(chat_id=1, requester_id=1, steps=[accion],
                           decision_id=42, execute_all=True)
        token = store.put(plan)

        recuperado = store.pop(token)

        assert recuperado.allow_modify is True
        assert recuperado.steps[0].request_intent is Intent.PREPARE_CALC
        assert recuperado.steps[0].request_params == {
            "formula": "Zr", "red_cristalina": "hcp",
        }
        assert recuperado.decision_id == 42
        assert recuperado.execute_all is True

    def test_sobrevive_al_reinicio_del_proceso(self, tmp_path):
        # El caso de la bitácora: el usuario ve la tarjeta, el bot se
        # reinicia, el usuario aprieta el botón. Antes eso daba "expiró"
        # con el TTL entero por delante.
        token = self._store(tmp_path).put(_plan())

        otro_proceso = self._store(tmp_path)

        assert otro_proceso.status(token) == "vigente"
        assert otro_proceso.pop(token) is not None

    def test_consumido_sigue_siendo_consumido_tras_reiniciar(self, tmp_path):
        # Doble toque con un reinicio en el medio: hay que poder decir "ya
        # se hizo", no "no lo conozco". Decirle a alguien que expiró algo
        # que en realidad se ejecutó lo manda a repetir un sbatch.
        store = self._store(tmp_path)
        token = store.put(_plan())
        store.pop(token)

        assert self._store(tmp_path).status(token) == "consumido"

    def test_dos_pop_simultaneos_ejecutan_una_sola_vez(self, tmp_path):
        # El lock de proceso del store en memoria no sirve si mañana corren
        # dos; acá lo garantiza el UPDATE condicional.
        import threading

        store = self._store(tmp_path)
        token = store.put(_plan())
        recuperados = []
        arranquen = threading.Event()

        def toque():
            arranquen.wait()
            recuperados.append(store.pop(token))

        hilos = [threading.Thread(target=toque) for _ in range(8)]
        for h in hilos:
            h.start()
        arranquen.set()
        for h in hilos:
            h.join()

        assert sum(1 for r in recuperados if r is not None) == 1


# ---------------------------------------------------------------------------
# SSH command construction (sin conexión real: interceptamos _run)
# ---------------------------------------------------------------------------


class RecordingGateway(SSHClusterGateway):
    """Reemplaza la ejecución real para capturar el comando construido."""

    def __init__(self):
        super().__init__(host="fake", user="fake", key_path="/dev/null")
        self.commands: list[str] = []

    def _run(self, command: str, *, reintentable: bool = False) -> CommandResult:
        # `reintentable` se acepta y se ignora: estos tests miden el
        # comando CONSTRUIDO, no la política de reintento (que se prueba
        # aparte, en tests/test_tolerancia_fallos.py).
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
            lambda cmd, **kw: CommandResult(ok=True, stdout="Submitted batch job 4242\n"),
        )
        result = gw.submit_job(SlurmJobRequest(script_path="/a/b.sh"))
        assert result.job_id == "4242"

    def test_submit_ok_but_unparsable_output_yields_no_job_id(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run", lambda cmd, **kw: CommandResult(ok=True, stdout="algo inesperado"),
        )
        result = gw.submit_job(SlurmJobRequest(script_path="/a/b.sh"))
        assert result.ok
        assert result.job_id is None

    def test_submit_failure_has_no_job_id(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run", lambda cmd, **kw: CommandResult(ok=False, stderr="sbatch: error"),
        )
        result = gw.submit_job(SlurmJobRequest(script_path="/a/b.sh"))
        assert not result.ok
        assert result.job_id is None

    def test_job_state_command_and_parsing(self, monkeypatch):
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run", lambda cmd, **kw: CommandResult(ok=True, stdout="COMPLETED\n"),
        )
        lectura = gw.job_state(JobId(value="4242"))
        assert lectura.state == "COMPLETED"
        assert lectura.reachable is True

    def test_job_state_uses_parsable_format(self):
        gw = RecordingGateway()
        gw.job_state(JobId(value="99"))
        cmd = gw.commands[0]
        assert "--parsable2" in cmd
        assert "--noheader" in cmd
        assert "sacct -j 99" in cmd

    def test_job_state_sin_transporte_no_es_alcanzable(self, monkeypatch):
        # No hubo conversación: el cluster no dijo nada sobre el trabajo.
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run",
            lambda cmd, **kw: CommandResult(
                ok=False, stderr="error", reason=CommandFailureReason.TRANSPORT
            ),
        )
        lectura = gw.job_state(JobId(value="1"))
        assert lectura.state is None
        assert lectura.reachable is False

    def test_job_state_con_salida_vacia_si_es_alcanzable(self, monkeypatch):
        # `sacct` contestó y no conoce el trabajo: eso SÍ es una respuesta,
        # y es la que hace que el monitor lo dé por perdido.
        gw = RecordingGateway()
        monkeypatch.setattr(gw, "_run", lambda cmd, **kw: CommandResult(ok=True, stdout=""))
        lectura = gw.job_state(JobId(value="1"))
        assert lectura.state is None
        assert lectura.reachable is True

    def test_job_state_comando_fallado_sigue_siendo_respuesta(self, monkeypatch):
        # `sacct` no existe, o permisos: el cluster habló, aunque para mal.
        # No es lo mismo que no poder preguntarle.
        gw = RecordingGateway()
        monkeypatch.setattr(
            gw, "_run",
            lambda cmd, **kw: CommandResult(
                ok=False, stderr="sacct: not found",
                reason=CommandFailureReason.COMMAND,
            ),
        )
        lectura = gw.job_state(JobId(value="1"))
        assert lectura.state is None
        assert lectura.reachable is True

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
        monkeypatch.setattr(gw, "_run", lambda cmd, **kw: CommandResult(ok=True, stdout=""))
        result = gw.make_directory("/home/user/pruebas")
        assert result.ok
        assert result.stdout == "Directorio listo: /home/user/pruebas"

    def test_make_directory_failure_passes_through(self, monkeypatch):
        gw = RecordingGateway()
        failure = CommandResult(ok=False, stderr="mkdir: permiso denegado")
        monkeypatch.setattr(gw, "_run", lambda cmd, **kw: failure)
        result = gw.make_directory("/root/prohibido")
        # El fallo llega intacto: la rama del mensaje fabricado no se activa.
        assert result is failure

    def test_list_directory_uses_tree_and_quotes_path(self):
        gw = RecordingGateway()
        gw.list_directory("/data/becario runs")
        assert gw.commands[0] == "tree -L 2 --noreport -- '/data/becario runs'"

    def test_list_directory_failure_passes_through(self, monkeypatch):
        # Un fallo real (no "command not found") no dispara el fallback.
        gw = RecordingGateway()
        failure = CommandResult(ok=False, stderr="tree: permiso denegado")
        monkeypatch.setattr(gw, "_run", lambda cmd, **kw: failure)
        assert gw.list_directory("/root/prohibido") is failure

    def test_list_directory_falls_back_to_find_without_tree(self):
        gw = RecordingGateway()
        outputs = {
            "tree": CommandResult(
                ok=False, stderr="bash: line 1: tree: command not found"
            ),
            "find": CommandResult(
                ok=True,
                stdout=(
                    "/data/runs/corrida_1\n"
                    "/data/runs/corrida_1/INCAR\n"
                    "/data/runs/corrida_2\n"
                ),
            ),
        }
        gw._run = lambda cmd, **kw: outputs["tree" if cmd.startswith("tree") else "find"]
        result = gw.list_directory("/data/runs")
        assert result.ok
        # El fallback dibuja ramas como `tree`, no indentación plana.
        assert result.stdout == (
            "/data/runs\n"
            "├── corrida_1\n"
            "│   └── INCAR\n"
            "└── corrida_2"
        )

    def test_list_directory_fallback_find_failure_passes_through(self):
        gw = RecordingGateway()
        find_failure = CommandResult(ok=False, stderr="find: no existe")
        gw._run = lambda cmd, **kw: (
            CommandResult(ok=False, stderr="tree: command not found")
            if cmd.startswith("tree")
            else find_failure
        )
        assert gw.list_directory("/data/nada") is find_failure

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


# ---------------------------------------------------------------------------
# ChatLog (bitácora de conversación)
# ---------------------------------------------------------------------------


class TestSQLiteChatLog:
    def test_schema_creation(self, tmp_path):
        import sqlite3

        path = str(tmp_path / "chat.db")
        SQLiteChatLogRepository(path)
        with sqlite3.connect(path) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "chat_messages" in names
        assert "idx_chat_messages_chat_fecha" in names

    def test_add_recent_roundtrip(self, tmp_path):
        repo = SQLiteChatLogRepository(str(tmp_path / "chat.db"))
        repo.add(chat_id=100, role="user", text="hola")
        repo.add(chat_id=100, role="bot", text="¿en qué te ayudo?")
        rows = repo.recent(100)
        assert [(r["role"], r["text"]) for r in rows] == [
            ("user", "hola"),
            ("bot", "¿en qué te ayudo?"),
        ]
        assert all(r["chat_id"] == 100 for r in rows)

    def test_created_at_is_utc_iso(self, tmp_path):
        from datetime import datetime, timezone

        repo = SQLiteChatLogRepository(str(tmp_path / "chat.db"))
        repo.add(chat_id=100, role="user", text="hola")
        created = datetime.fromisoformat(repo.recent(100)[0]["created_at"])
        assert created.tzinfo is not None
        assert created.utcoffset().total_seconds() == 0
        assert abs((datetime.now(timezone.utc) - created).total_seconds()) < 60

    def test_role_constraint(self, tmp_path):
        import sqlite3

        repo = SQLiteChatLogRepository(str(tmp_path / "chat.db"))
        with pytest.raises(sqlite3.IntegrityError):
            repo.add(chat_id=100, role="sistema", text="rol inválido")

    def test_recent_returns_newest_last_with_limit(self, tmp_path):
        repo = SQLiteChatLogRepository(str(tmp_path / "chat.db"))
        for i in range(10):
            repo.add(chat_id=100, role="user", text=f"mensaje {i}")
        rows = repo.recent(100, limit=3)
        # Los 3 más nuevos, del más viejo al más nuevo.
        assert [r["text"] for r in rows] == ["mensaje 7", "mensaje 8", "mensaje 9"]

    def test_chats_are_isolated(self, tmp_path):
        repo = SQLiteChatLogRepository(str(tmp_path / "chat.db"))
        repo.add(chat_id=100, role="user", text="del chat 100")
        repo.add(chat_id=200, role="user", text="del chat 200")
        assert [r["text"] for r in repo.recent(100)] == ["del chat 100"]
        assert [r["text"] for r in repo.recent(200)] == ["del chat 200"]

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "chat.db")
        SQLiteChatLogRepository(path).add(chat_id=100, role="user", text="hola")
        rows = SQLiteChatLogRepository(path).recent(100)
        assert [r["text"] for r in rows] == ["hola"]


class TestSQLiteRouterDecisionLog:
    def _log(self, tmp_path):
        return SQLiteRouterDecisionLog(
            str(tmp_path / "decisiones.db"), model="qwen2.5:7b"
        )

    def test_add_returns_id_and_defaults_to_routed(self, tmp_path):
        log = self._log(tmp_path)
        steps = json.dumps([{"action": "crear_directorio", "parametros": {}}])
        decision_id = log.add(
            chat_id=100, user_id=1, text="creame la carpeta pruebas",
            steps_json=steps, latency_seconds=5.4,
        )
        rows = log.rows()
        assert rows[0]["id"] == decision_id
        assert rows[0]["outcome"] == "routed"
        assert rows[0]["model"] == "qwen2.5:7b"
        assert rows[0]["latency_seconds"] == pytest.approx(5.4)
        assert json.loads(rows[0]["steps_json"])[0]["action"] == "crear_directorio"

    def test_set_outcome_updates_only_that_row(self, tmp_path):
        log = self._log(tmp_path)
        steps = json.dumps([])
        first = log.add(100, 1, "enviá el cálculo", steps, 1.0)
        second = log.add(100, 1, "cancelá el 42", steps, 1.0)
        log.set_outcome(first, "confirmed")
        by_id = {r["id"]: r["outcome"] for r in log.rows()}
        assert by_id == {first: "confirmed", second: "routed"}

    def test_rows_filters_by_outcome(self, tmp_path):
        log = self._log(tmp_path)
        steps = json.dumps([])
        first = log.add(100, 1, "uno", steps, 1.0)
        log.add(100, 1, "dos", steps, 1.0)
        log.set_outcome(first, "cancelled")
        assert [r["text"] for r in log.rows(outcome="cancelled")] == ["uno"]
        assert [r["text"] for r in log.rows(outcome="routed")] == ["dos"]

    def test_outcome_constraint(self, tmp_path):
        import sqlite3

        log = self._log(tmp_path)
        decision_id = log.add(100, 1, "hola", json.dumps([]), 1.0)
        with pytest.raises(sqlite3.IntegrityError):
            log.set_outcome(decision_id, "desenlace_invalido")

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "decisiones.db")
        SQLiteRouterDecisionLog(path, model="m").add(100, 1, "hola", "[]", 1.0)
        assert len(SQLiteRouterDecisionLog(path, model="m").rows()) == 1


# ---------------------------------------------------------------------------
# Concurrencia SQLite (WAL + busy_timeout compartidos por todos los repos)
# ---------------------------------------------------------------------------


class TestSQLiteConcurrency:
    """Todos los repositorios comparten un archivo: sin WAL ni busy_timeout,
    escritores concurrentes fallan con «database is locked». Estos tests
    verifican la configuración y el comportamiento bajo escritura intercalada.

    Determinismo: una barrera arranca los hilos a la vez y cada uno escribe
    una cantidad fija de filas; no hay sleeps como sincronización."""

    def test_connections_use_wal_and_busy_timeout(self, tmp_path):
        import sqlite3

        path = str(tmp_path / "shared.db")
        repo = SQLiteChatLogRepository(path)
        conn = repo._connect()
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        finally:
            conn.close()
        # WAL es persistente: una conexión nueva sobre el mismo archivo
        # (p. ej. de otro repositorio) también lo hereda.
        with sqlite3.connect(path) as raw:
            assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    @staticmethod
    def _run_writers(*writers):
        """Lanza un hilo por writer, sincronizados con una barrera, y
        devuelve las excepciones capturadas (la lista debe quedar vacía)."""
        import threading

        barrier = threading.Barrier(len(writers))
        errors: list[Exception] = []

        def _wrap(write):
            def _target():
                barrier.wait()
                try:
                    write()
                except Exception as exc:  # noqa: BLE001 - el test la reporta
                    errors.append(exc)

            return _target

        threads = [threading.Thread(target=_wrap(w)) for w in writers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errors

    def test_two_instances_write_interleaved_without_lock_errors(self, tmp_path):
        path = str(tmp_path / "shared.db")
        repo_a = SQLiteChatLogRepository(path)
        repo_b = SQLiteChatLogRepository(path)
        n = 50

        def writer(repo, role):
            def write():
                for i in range(n):
                    repo.add(chat_id=1, role=role, text=f"{role}-{i}")

            return write

        errors = self._run_writers(writer(repo_a, "user"), writer(repo_b, "bot"))

        assert errors == []
        rows = repo_a.recent(1, limit=2 * n + 1)
        assert len(rows) == 2 * n
        # Ninguna escritura se perdió: están las n de cada instancia.
        texts = {r["text"] for r in rows}
        assert texts == {f"user-{i}" for i in range(n)} | {f"bot-{i}" for i in range(n)}

    def test_distinct_repositories_share_the_file_without_lock_errors(self, tmp_path):
        # Réplica del escenario de producción: bitácora de chat e historial
        # escribiendo a la vez sobre el mismo archivo.
        path = str(tmp_path / "shared.db")
        chat = SQLiteChatLogRepository(path)
        history = SQLiteHistoryRepository(path)
        history.ensure_schema()
        # 25 por hilo: HistoryFilter acota limit a 50 y acá se verifica el
        # conteo exacto pidiendo n + 1.
        n = 25

        def write_chat():
            for i in range(n):
                chat.add(chat_id=1, role="user", text=f"msg-{i}")

        def write_history():
            for i in range(n):
                history.add(owner_id=7, job_id=str(i), nombre_trabajo=f"job-{i}", estado="PENDING")

        errors = self._run_writers(write_chat, write_history)

        assert errors == []
        assert len(chat.recent(1, limit=n + 1)) == n
        assert len(history.search(HistoryFilter(owner_id=7, limit=n + 1))) == n


class TestFormatTree:
    """`_format_tree` emula `tree -L 2` con ramas cuando el cluster no
    tiene el binario `tree` (caso real: el contenedor SLURM local)."""

    def test_draws_branches_like_tree(self):
        from becario.infrastructure.ssh_gateway import _format_tree

        find_out = "\n".join([
            "/root/becario_runs/W",
            "/root/becario_runs/W/bcc",
            "/root/becario_runs/W/fcc",
            "/root/becario_runs/Zr",
            "/root/becario_runs/Zr/hcp",
        ])
        assert _format_tree("/root/becario_runs", find_out) == "\n".join([
            "/root/becario_runs",
            "├── W",
            "│   ├── bcc",
            "│   └── fcc",
            "└── Zr",
            "    └── hcp",
        ])

    def test_empty_listing_shows_only_base(self):
        from becario.infrastructure.ssh_gateway import _format_tree

        assert _format_tree("/data/x", "") == "/data/x"

    def test_single_level(self):
        from becario.infrastructure.ssh_gateway import _format_tree

        out = _format_tree("/base", "/base/solo")
        assert out == "/base\n└── solo"


class TestSQLitePendingEditStore:
    """Pedidos esperando respuesta, en disco.

    El caso que lo justifica es el de la bitácora: el bot pregunta con qué
    fase de ZrO2 seguir, el usuario contesta «tetragonal» trece minutos
    después —con un TTL de treinta— y recibe «No pude interpretar tu
    pedido». No había vencido: el proceso se reinició y el pendiente vivía
    en un `dict`.
    """

    def _store(self, tmp_path) -> SQLitePendingEditStore:
        return SQLitePendingEditStore(str(tmp_path / "becario.db"))

    def _edit(self, chat_id: int = 77, awaiting_index=None) -> PendingEdit:
        return PendingEdit(
            steps=[(Intent.PREPARE_CALC, {"formula": "ZrO2", "tipo_calculo": "relajacion"})],
            chat_id=chat_id,
            awaiting_index=awaiting_index,
        )

    def test_round_trip_conserva_pasos_intent_y_hueco(self, tmp_path):
        store = self._store(tmp_path)
        store.put(5, self._edit(awaiting_index=2))

        recuperado = store.get(5)

        assert recuperado.steps == [
            (Intent.PREPARE_CALC, {"formula": "ZrO2", "tipo_calculo": "relajacion"})
        ]
        assert recuperado.awaiting_index == 2
        assert recuperado.chat_id == 77

    def test_sobrevive_al_reinicio_del_proceso(self, tmp_path):
        # El «tetragonal» de la bitácora, en un test.
        self._store(tmp_path).put(5, self._edit())

        recuperado = self._store(tmp_path).get(5)

        assert recuperado is not None
        assert recuperado.steps[0][1]["formula"] == "ZrO2"

    def test_get_no_consume_y_pop_si(self, tmp_path):
        store = self._store(tmp_path)
        store.put(5, self._edit())

        assert store.get(5) is not None
        assert store.has(5) is True
        assert store.pop(5) is not None
        assert store.get(5) is None
        assert store.has(5) is False

    def test_un_pedido_nuevo_pisa_al_anterior(self, tmp_path):
        # Uno por usuario: quien contesta una repregunta contesta la última.
        store = self._store(tmp_path)
        store.put(5, self._edit())
        store.put(5, PendingEdit(steps=[(Intent.LIST_FILES, {})], chat_id=99))

        recuperado = store.get(5)

        assert recuperado.steps[0][0] is Intent.LIST_FILES
        assert recuperado.chat_id == 99

    def test_pop_de_un_vencido_lo_devuelve_para_poder_avisar(self, tmp_path):
        # Vencido igual se saca: el servicio necesita su contenido para
        # decir QUÉ se venció, en vez de dejar al usuario esperando.
        store = self._store(tmp_path)
        viejo = self._edit()
        viejo.created_at -= 5000
        store.put(5, viejo)

        recuperado = store.pop(5)

        assert recuperado is not None
        assert recuperado.steps[0][1]["formula"] == "ZrO2"

    def test_pop_expired_saca_solo_los_vencidos(self, tmp_path):
        store = self._store(tmp_path)
        vencido = self._edit(chat_id=11)
        vencido.created_at -= 5000
        store.put(1, vencido)
        store.put(2, self._edit(chat_id=22))

        vencidos = store.pop_expired(ttl_seconds=1800)

        assert [(u, e.chat_id) for u, e in vencidos] == [(1, 11)]
        assert store.has(1) is False
        assert store.has(2) is True

    def test_una_fila_ilegible_no_tira_abajo_la_conversacion(self, tmp_path):
        # Perder un pendiente es malo; propagar la excepción hasta el
        # handler y dejar al usuario sin ninguna respuesta, peor.
        import sqlite3 as _sq

        store = self._store(tmp_path)
        store.put(5, self._edit())
        with _sq.connect(str(tmp_path / "becario.db")) as conn:
            conn.execute("UPDATE pedidos_pendientes SET edit_json = '{no json'")

        assert store.get(5) is None
