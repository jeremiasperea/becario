"""Tests del generador de inputs VASP (usa ASE de verdad, sin cluster)."""
from pathlib import Path

import pytest
from ase import Atoms

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


class TestIncarQualityTags:
    def test_common_quality_tags_present(self, generator):
        req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        for tag in ("ADDGRID = .TRUE.", "LASPH = .TRUE.", "LMAXMIX = 6",
                    "ALGO = Fast", "NELMIN = 10"):
            assert tag in incar

    def test_lreal_false_not_auto(self, generator):
        """Desviación deliberada de la referencia: celdas chicas => .FALSE."""
        req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "LREAL = .FALSE." in incar

    def test_metal_smearing_not_tetrahedron(self, generator):
        """ISMEAR=1 (metales), nunca -5 al relajar."""
        req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "ISMEAR = 1" in incar
        assert "ISMEAR = -5" not in incar


class TestNbands:
    def test_computed_from_zval_table(self, generator):
        """Zr hcp: 2 átomos * ZVAL 12 = 24 e-; NBANDS = ceil(24/2 * 1.2) = 15."""
        req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "NBANDS = 15" in incar

    def test_absent_when_element_not_in_table(self, generator):
        """Un elemento sin ZVAL en la tabla => VASP calcula NBANDS."""
        req = VaspCalcRequest(formula="Al", crystal="fcc", calc_kind=CalcKind.STATIC)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "NBANDS" not in incar

    def test_explicit_override_wins(self, generator):
        """Un NBANDS pedido a mano se respeta aun sin ZVAL del elemento."""
        req = VaspCalcRequest(formula="Al", crystal="fcc", nbands=200)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "NBANDS = 200" in incar


class TestIsif:
    def test_relax_override_relaxes_ions_only(self, generator):
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.RELAX, isif=2)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "ISIF = 2" in incar

    def test_static_ignores_isif(self, generator):
        """ISIF no tiene sentido con NSW=0: no se emite ni forzándolo."""
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.STATIC, isif=3)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "ISIF" not in incar


class TestProvidedAtoms:
    """El seam MP: si se pasan átomos ya resueltos, el generador los usa tal
    cual (no arma la estructura desde la fórmula del pedido)."""

    def test_generate_uses_provided_atoms_verbatim(self, generator):
        atoms = Atoms(
            "FeO",
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[4.3, 4.3, 4.3],
            pbc=True,
        )
        # el req dice Zr, pero los átomos provistos (FeO) son los que mandan
        req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23)
        result = generator.generate(req, atoms)
        assert sorted(result.elements) == ["Fe", "O"]
        poscar = (Path(result.local_dir) / "POSCAR").read_text()
        assert "Fe" in poscar and "O" in poscar

    def test_generate_without_atoms_builds_from_formula(self, generator):
        # sin átomos: comportamiento existente (ASE desde la fórmula)
        req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23)
        result = generator.generate(req)
        assert result.elements == ["Zr"]


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


class TestRunDirCollision:
    """El sello de tiempo tiene resolución de UN SEGUNDO. Dos cálculos del
    mismo material y tipo pedidos juntos —un plan multi-paso, o dos pasos de
    un batch— caían en el mismo nombre y el segundo moría con FileExistsError.
    """

    def _req(self):
        return VaspCalcRequest(formula="ZrO2", crystal="fluorite", lattice_a=5.14)

    def test_two_identical_runs_in_the_same_second_both_succeed(self, generator):
        first = generator.generate(self._req())
        second = generator.generate(self._req())
        assert first.local_dir != second.local_dir
        assert Path(first.local_dir).exists() and Path(second.local_dir).exists()

    def test_suffix_keeps_the_chronological_prefix(self, generator):
        """El sufijo va al final: el nombre sigue ordenando por fecha, y el
        prefijo `{formula}_{tipo}_` queda intacto."""
        first = generator.generate(self._req())
        second = generator.generate(self._req())
        assert second.run_name.startswith("ZrO2_estatico_")
        assert second.run_name == f"{first.run_name}_2"

    def test_existing_run_is_never_overwritten(self, generator, tmp_path):
        """La propiedad que hay que conservar: nunca se escribe encima."""
        first = generator.generate(self._req())
        marker = Path(first.local_dir) / "NO_ME_PISES"
        marker.write_text("resultado previo")
        generator.generate(self._req())
        assert marker.read_text() == "resultado previo"

    def test_gives_up_loudly_instead_of_looping_forever(self, generator, monkeypatch):
        import becario.infrastructure.vasp_inputs as vi

        monkeypatch.setattr(vi, "_MAX_RUN_DIR_ATTEMPTS", 2)
        generator.generate(self._req())
        generator.generate(self._req())
        with pytest.raises(RuntimeError, match="directorio de corrida"):
            generator.generate(self._req())
