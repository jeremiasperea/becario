"""Tests del arranque desde una estructura ya relajada (CONTCAR previo).

Todo lo de acá es fail-closed: cuando algo no está en condiciones, lo que se
verifica es que NO se devuelva una estructura. El caso que justifica el
módulo entero es el último grupo: una relajación que terminó con código 0,
que Slurm reporta COMPLETED, cuyo CONTCAR abre perfecto, y que NO convergió.
"""
from __future__ import annotations

import pytest
from ase.build import bulk
from ase.io import write

from becario.application.relaxed_source import (
    RelaxedSourceError,
    resolve_relaxed_structure,
)

OWNER = 111
RUN_DIR = "/home/alice/becario_runs/Zr_relajacion_20260101_101010"


def _contcar_text(a=3.23):
    import io

    buf = io.StringIO()
    write(buf, bulk("Zr", "hcp", a=a), format="vasp", direct=True)
    return buf.getvalue()


def _oszicar(steps: int) -> str:
    return "".join(
        f"       N       E                     dE\n"
        f"{i:5d} F= -.31E+02 E0= -.31E+02  d E =-.31E+02\n"
        for i in range(1, steps + 1)
    )


class FakeCalcRuns:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [{
            "job_id": "12345", "job_name": "Zr_relajacion", "run_dir": RUN_DIR,
        }]
        self.queries: list[tuple] = []

    def find_recent(self, owner_id, job_name_prefix="", limit=5):
        self.queries.append((owner_id, job_name_prefix, limit))
        return list(self.rows)


class FakeCluster:
    """Cluster con archivos en memoria. `state` es lo que devuelve sacct."""

    def __init__(self, files=None, state="COMPLETED"):
        self.files = files if files is not None else {
            f"{RUN_DIR}/CONTCAR": _contcar_text(),
            f"{RUN_DIR}/INCAR": "NSW = 180\nIBRION = 2\n",
            f"{RUN_DIR}/OSZICAR": _oszicar(37),
        }
        self.state = state

    def job_state(self, job_id):
        return self.state

    def file_exists(self, path):
        return path in self.files

    def read_file(self, path, max_bytes=None):
        return self.files.get(path)


def _resolve(runs=None, cluster=None, formula="Zr"):
    return resolve_relaxed_structure(
        runs or FakeCalcRuns(), cluster or FakeCluster(), OWNER, formula
    )


class TestHappyPath:
    def test_returns_the_relaxed_atoms(self):
        result = _resolve()
        assert result.atoms.get_chemical_formula() == "Zr2"
        assert result.converged is True
        assert result.warning == ""

    def test_looks_up_only_this_users_relaxations(self):
        """La estructura de partida sale de las corridas de QUIEN pide."""
        runs = FakeCalcRuns()
        _resolve(runs=runs, formula="ZrO2")
        owner, prefix, _ = runs.queries[0]
        assert owner == OWNER
        assert prefix == "ZrO2_relajacion"

    def test_note_names_the_run_it_started_from(self):
        note = _resolve().note()
        assert "Zr_relajacion" in note and RUN_DIR in note


class TestFailsClosed:
    """Los cuatro casos que el usuario pidió que se rechacen de plano."""

    def test_no_previous_relaxation(self):
        with pytest.raises(RelaxedSourceError, match="No encontré ninguna relajación"):
            _resolve(runs=FakeCalcRuns(rows=[]))

    def test_job_still_running(self):
        with pytest.raises(RelaxedSourceError, match="todavía está corriendo"):
            _resolve(cluster=FakeCluster(state="RUNNING"))

    def test_job_still_queued(self):
        with pytest.raises(RelaxedSourceError, match="en cola"):
            _resolve(cluster=FakeCluster(state="PENDING"))

    @pytest.mark.parametrize("state", ["FAILED", "TIMEOUT", "CANCELLED by 42", "NODE_FAIL"])
    def test_job_ended_badly(self, state):
        with pytest.raises(RelaxedSourceError):
            _resolve(cluster=FakeCluster(state=state))

    def test_contcar_missing(self):
        cluster = FakeCluster()
        del cluster.files[f"{RUN_DIR}/CONTCAR"]
        with pytest.raises(RelaxedSourceError, match="no tiene CONTCAR"):
            _resolve(cluster=cluster)

    def test_contcar_empty(self):
        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/CONTCAR"] = "   \n"
        with pytest.raises(RelaxedSourceError, match="vacío"):
            _resolve(cluster=cluster)

    def test_contcar_truncated(self):
        """Un CONTCAR a medio escribir: parsea mal, no se usa."""
        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/CONTCAR"] = "Zr\n 1.0\n  3.23 0.0"
        with pytest.raises(RelaxedSourceError, match="incompleto o"):
            _resolve(cluster=cluster)

    def test_unknown_slurm_state_does_not_block(self):
        """Slurm se olvida de las corridas viejas: eso no es un fallo, y los
        chequeos de archivo alcanzan para decidir."""
        assert _resolve(cluster=FakeCluster(state="")).atoms is not None


class TestConvergenceWarnsButDoesNotBlock:
    """El caso peligroso: terminó bien para Slurm y NO está relajada.

    Acá NO se bloquea — se avisa y decide el usuario, que es lo que pidió.
    """

    def test_exhausted_nsw_is_flagged(self):
        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/OSZICAR"] = _oszicar(180)  # llegó al tope
        result = _resolve(cluster=cluster)
        assert result.converged is False
        assert result.atoms is not None, "avisa, pero NO bloquea"
        assert "180 pasos" in result.warning
        assert "NO está relajada" in result.warning

    def test_warning_travels_in_the_note(self):
        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/OSZICAR"] = _oszicar(180)
        assert "OJO" in _resolve(cluster=cluster).note()

    def test_finishing_early_means_it_converged(self):
        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/OSZICAR"] = _oszicar(37)
        assert _resolve(cluster=cluster).converged is True

    def test_unverifiable_convergence_says_so(self):
        cluster = FakeCluster()
        del cluster.files[f"{RUN_DIR}/OSZICAR"]
        result = _resolve(cluster=cluster)
        assert result.converged is None
        assert "No pude verificar" in result.warning
        assert result.atoms is not None

    def test_incar_without_nsw_is_unverifiable(self):
        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/INCAR"] = "IBRION = 2\n"
        assert _resolve(cluster=cluster).converged is None


class TestTruncatedOszicarIsNotMistakenForConvergence:
    """`read_file` baja un PREFIJO. Si el OSZICAR se corta, los pasos que
    faltan son los del final, y contar sobre el prefijo daría MENOS pasos que
    NSW — o sea "convergió" para una corrida que no convergió. El error tiene
    que caer del lado seguro."""

    def test_truncated_oszicar_is_unverifiable_not_converged(self):
        from becario.application.relaxed_source import _OSZICAR_MAX_BYTES

        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/OSZICAR"] = "x" * _OSZICAR_MAX_BYTES
        result = _resolve(cluster=cluster)
        assert result.converged is None, "no puede afirmar que convergió"
        assert "demasiado grande" in result.warning
        assert result.atoms is not None  # sigue sin bloquear
