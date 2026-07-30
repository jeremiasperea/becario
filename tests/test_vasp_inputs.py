"""Tests del generador de inputs VASP (usa ASE de verdad, sin cluster)."""
import logging
from pathlib import Path

import pytest
from ase import Atoms
from pydantic import ValidationError

from becario.domain.models import CalcKind, StructureKind, VaspCalcRequest
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
        assert "NSW = 180" in incar

    def test_relax_uses_group_force_criterion(self, generator):
        """EDIFFG=-0.02 eV/Å es el criterio del grupo, no el -0.01 del andamio."""
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.RELAX)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "EDIFFG = -0.02" in incar

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


class TestNsw:
    """El presupuesto de pasos iónicos lo fija el TIPO de cálculo; el pedido
    solo puede afinarlo donde tiene sentido (los tipos que relajan)."""

    def test_relax_default_is_group_production_value(self, generator):
        """180, no 60: un presupuesto corto termina con código 0 y deja un
        CONTCAR a medio relajar que parece válido."""
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.RELAX)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "NSW = 180" in incar

    def test_relax_override_wins(self, generator):
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.RELAX, nsw=300)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "NSW = 300" in incar

    def test_static_forces_zero_even_when_nsw_requested(self, generator):
        """El tipo manda: un estático con pasos iónicos no es un estático.

        Y al forzarlo (en vez de rechazarlo) queda inalcanzable un INCAR con
        NSW>1 sin IBRION, que VASP leería como dinámica molecular."""
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.STATIC, nsw=180)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "NSW = 0" in incar
        assert "IBRION" not in incar
        assert "EDIFFG" not in incar
        assert "ISIF" not in incar

    def test_forcing_is_logged_not_silent(self, generator, caplog):
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.STATIC, nsw=180)
        with caplog.at_level(logging.WARNING):
            generator.generate(req)
        assert "NSW=180 ignorado" in caplog.text

    def test_encut_scan_points_have_no_ionic_steps(self, generator):
        req = VaspCalcRequest(
            formula="W", calc_kind=CalcKind.ENCUT_SCAN, encut_values=[300, 400]
        )
        run_dir = Path(generator.generate(req).local_dir)
        for encut in (300, 400):
            incar = (run_dir / f"encut_{encut}" / "INCAR").read_text()
            assert "NSW = 0" in incar
            assert "IBRION" not in incar

    def test_relax_with_zero_nsw_is_rejected(self):
        """Caso espejo: una `relajacion` sin pasos iónicos sería un estático
        con el nombre equivocado. El cero se pide eligiendo el otro tipo."""
        with pytest.raises(ValidationError, match="NSW>=1"):
            VaspCalcRequest(formula="W", calc_kind=CalcKind.RELAX, nsw=0)

    def test_static_with_zero_nsw_is_consistent_not_an_error(self, generator):
        """Pedir explícitamente NSW=0 en un estático coincide con el default:
        no hay nada que forzar ni que avisar."""
        req = VaspCalcRequest(formula="W", calc_kind=CalcKind.STATIC, nsw=0)
        incar = (Path(generator.generate(req).local_dir) / "INCAR").read_text()
        assert "NSW = 0" in incar


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


class TestSlabCalc:
    """El generador tiene su PROPIO camino de construcción: si la losa no
    llega hasta acá, "armá el slab y mandalo a relajar" produce inputs de
    bulk sin que nada avise."""

    def _req(self, **kw):
        base = dict(
            formula="ZrO2", kind=StructureKind.SLAB, crystal="fluorite",
            lattice_a=5.14, miller=(0, 0, 1), layers=4,
        )
        base.update(kw)
        return VaspCalcRequest(**base)

    def test_slab_poscar_is_not_a_bulk_poscar(self, generator):
        # calc_kind distinto a propósito: el nombre del directorio de corrida
        # tiene resolución de un segundo y `mkdir(exist_ok=False)` choca si dos
        # corridas del mismo material y tipo caen en el mismo segundo.
        slab = generator.generate(self._req())
        bulk_ = generator.generate(
            VaspCalcRequest(
                formula="ZrO2", crystal="fluorite", lattice_a=5.14,
                calc_kind=CalcKind.RELAX,
            )
        )
        assert slab.n_atoms > bulk_.n_atoms
        assert "c=" in slab.cell_summary

    def test_slab_keeps_vacuum_in_the_generated_cell(self, generator):
        result = generator.generate(self._req())
        c = float(result.cell_summary.split("c=")[1].split(" ")[0])
        assert c > 30  # espesor + 2 x 15 Å de vacío

    def test_kpoints_collapse_along_the_vacuum(self, generator):
        """La grilla automática es n_i = round(30/a_i): sobre el eje con
        vacío el parámetro es enorme, así que ahí tiene que dar 1 — muestrear
        el vacío sería tirar tiempo de cómputo."""
        result = generator.generate(self._req())
        assert result.kpoints[2] == 1
        assert all(k > 1 for k in result.kpoints[:2])

    def test_molecule_kind_rejected_for_vasp(self):
        with pytest.raises(ValidationError, match="moléculas"):
            VaspCalcRequest(formula="H2O", kind=StructureKind.MOLECULE)


class TestSlabFromProvidedCell:
    """Una estructura ya resuelta (Materials Project o CONTCAR relajado) es
    la celda de PARTIDA de la losa, no el resultado.

    Regresión: antes se devolvía esa celda entera sin cortar — un bulk con
    nombre de slab. Afectaba a todo compuesto, porque `source=auto` los manda
    a Materials Project."""

    def _slab_req(self):
        return VaspCalcRequest(
            formula="ZrO2", kind=StructureKind.SLAB, crystal="fluorite",
            lattice_a=5.14, miller=(0, 0, 1), layers=5,
        )

    def test_surface_is_actually_cut_from_the_provided_cell(self, generator, tmp_path):
        from ase.build import bulk

        provided = bulk("ZrO2", "fluorite", a=5.14, cubic=True)
        result = generator.generate(self._slab_req(), atoms=provided)
        assert result.n_atoms > len(provided), "la celda entró sin cortar"
        c = float(result.cell_summary.split("c=")[1].split(" ")[0])
        assert c > 30, "sin vacío no es una superficie"

    def test_matches_the_slab_built_from_the_formula(self, generator, tmp_path):
        """Partir de la misma celda convencional tiene que dar la misma losa."""
        from ase.build import bulk

        other = VaspInputGenerator(workdir=str(tmp_path / "otro"))
        from_formula = generator.generate(self._slab_req())
        from_cell = other.generate(
            self._slab_req(), atoms=bulk("ZrO2", "fluorite", a=5.14, cubic=True)
        )
        assert from_cell.n_atoms == from_formula.n_atoms
        assert from_cell.cell_summary == from_formula.cell_summary

    def test_bulk_with_provided_cell_is_untouched(self, generator):
        from ase.build import bulk

        provided = bulk("ZrO2", "fluorite", a=5.14, cubic=True)
        result = generator.generate(
            VaspCalcRequest(formula="ZrO2", crystal="fluorite", lattice_a=5.14),
            atoms=provided,
        )
        assert result.n_atoms == len(provided)


class TestVacuumOnlyMakesSenseForSlabs:
    def test_bulk_calc_rejects_vacuum_instead_of_dropping_it(self):
        """El generador solo aplica vacío al cortar una losa: en un bulk lo
        descartaría en silencio, así que el pedido se rechaza."""
        with pytest.raises(ValidationError, match="solo aplica a una losa"):
            VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23, vacuum=12.0)
