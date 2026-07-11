"""Tests del generador de inputs VASP (usa ASE de verdad, sin cluster)."""
from pathlib import Path

import pytest

from becario.domain.models import CalcKind, VaspCalcRequest
from becario.infrastructure.vasp_inputs import VaspInputGenerator


@pytest.fixture()
def generator(tmp_path):
    return VaspInputGenerator(
        workdir=str(tmp_path),
        vasp_cmd="mpirun -np 8 vasp_std",
        vasp_prelude="export LD_LIBRARY_PATH=/data/intel-runtime/lib",
    )


class TestSingleCalc:
    def test_static_generates_complete_run_dir(self, generator):
        req = VaspCalcRequest(formula="Si", calc_kind=CalcKind.STATIC, encut=400)
        result = generator.generate(req)
        run_dir = Path(result.local_dir)
        for name in ("POSCAR", "INCAR", "KPOINTS", "run_vasp.sh"):
            assert (run_dir / name).exists()

        incar = (run_dir / "INCAR").read_text()
        assert "ENCUT = 400" in incar
        assert "NSW = 0" in incar
        assert "ISIF" not in incar

    def test_relax_uses_isif3(self, generator):
        """ISIF=3 es lo que minimiza los parámetros de red."""
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.RELAX)
        result = generator.generate(req)
        incar = (Path(result.local_dir) / "INCAR").read_text()
        assert "ISIF = 3" in incar
        assert "IBRION = 2" in incar
        assert "NSW = 60" in incar

    def test_job_script_cds_to_own_dir_with_prelude(self, generator):
        req = VaspCalcRequest(formula="Si")
        result = generator.generate(req)
        script = (Path(result.local_dir) / "run_vasp.sh").read_text()
        assert 'cd "$(dirname "$0")"' in script
        assert "export LD_LIBRARY_PATH=/data/intel-runtime/lib" in script
        assert "mpirun -np 8 vasp_std > vasp.out" in script

    def test_result_metadata(self, generator):
        req = VaspCalcRequest(formula="Si", supercell=(2, 2, 2))
        result = generator.generate(req)
        assert result.elements == ["Si"]
        assert result.n_atoms == 16  # 2 átomos/celda primitiva * 8
        assert all(k >= 1 for k in result.kpoints)
        assert result.encut_values == [520]

    def test_hcp_with_lattice_parameter(self, generator):
        req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23)
        result = generator.generate(req)
        assert result.n_atoms == 2
        assert "a=3.230" in result.cell_summary


class TestEncutScan:
    def test_scan_creates_one_subdir_per_point(self, generator):
        req = VaspCalcRequest(
            formula="Zr",
            calc_kind=CalcKind.ENCUT_SCAN,
            encut_values=[250, 300, 350],
        )
        result = generator.generate(req)
        run_dir = Path(result.local_dir)
        assert result.encut_values == [250, 300, 350]
        for encut in (250, 300, 350):
            subdir = run_dir / f"encut_{encut}"
            assert (subdir / "POSCAR").exists()
            assert (subdir / "KPOINTS").exists()
            assert f"ENCUT = {encut}" in (subdir / "INCAR").read_text()
        # La raíz no tiene inputs propios: solo el script (el POTCAR se arma
        # en el cluster).
        assert not (run_dir / "POSCAR").exists()
        assert (run_dir / "run_vasp.sh").exists()

    def test_scan_script_loops_over_points(self, generator):
        req = VaspCalcRequest(formula="Zr", calc_kind=CalcKind.ENCUT_SCAN)
        result = generator.generate(req)
        script = (Path(result.local_dir) / "run_vasp.sh").read_text()
        assert "for d in encut_*/" in script
        assert 'cp POTCAR "$d"' in script

    def test_scan_without_values_uses_default_range(self, generator):
        req = VaspCalcRequest(formula="Zr", calc_kind=CalcKind.ENCUT_SCAN)
        result = generator.generate(req)
        assert result.encut_values == list(range(300, 651, 50))


class TestRequestValidation:
    def test_scan_needs_at_least_two_values(self):
        with pytest.raises(ValueError):
            VaspCalcRequest(formula="Zr", encut_values=[400])

    def test_scan_values_out_of_range(self):
        with pytest.raises(ValueError):
            VaspCalcRequest(formula="Zr", encut_values=[50, 5000])

    def test_scan_values_are_sorted_and_deduplicated(self):
        req = VaspCalcRequest(formula="Zr", encut_values=[400, 300, 400])
        assert req.encut_values == [300, 400]

    def test_evil_formula_rejected(self):
        with pytest.raises(ValueError):
            VaspCalcRequest(formula="Zr; rm -rf /")

    def test_evil_time_limit_rejected(self):
        with pytest.raises(ValueError):
            VaspCalcRequest(formula="Zr", time_limit="1h; reboot")
