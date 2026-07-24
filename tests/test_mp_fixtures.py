"""PR3: adaptador MP contra fixtures serializados + integración con el generador.

Cubre R2 (fórmula), R3 (chemsys + filtro + alternativas), R4 (mp-id), R9
(seam: átomos de MP -> pipeline idéntico) y R7 (sin match).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from becario.domain.models import (
    CalcKind,
    StructureQuery,
    StructureResolutionError,
    StructureResolutionReason,
    VaspCalcRequest,
)
from becario.infrastructure.materials_project import MaterialsProjectProvider
from becario.infrastructure.vasp_inputs import VaspInputGenerator

from .mp_fixtures import FixtureRester, fe_o_docs


def _provider() -> MaterialsProjectProvider:
    return MaterialsProjectProvider(
        api_key="dummy", rester_factory=lambda: FixtureRester(fe_o_docs())
    )


class TestFormulaFixture:
    def test_formula_selects_correct_material(self):
        res = _provider().resolve(StructureQuery(formula="Fe2O3"))
        assert res.mp_id == "mp-19770"
        assert res.formula == "Fe2O3"
        assert res.spacegroup  # grupo espacial derivado de la estructura real

    def test_formula_no_match_is_no_match(self):
        with pytest.raises(StructureResolutionError) as exc:
            _provider().resolve(StructureQuery(formula="Zr2O3"))
        assert exc.value.reason is StructureResolutionReason.NO_MATCH


class TestChemsysFixture:
    def test_filters_pure_metal_and_picks_most_stable(self):
        res = _provider().resolve(
            StructureQuery(elements=("Fe", "O"), qualifier="O")
        )
        # Fe2O3 (e_hull 0.0) es el óxido más estable; el Fe puro (mp-13) se filtra
        assert res.mp_id == "mp-19770"
        alt_ids = {a.mp_id for a in res.alternatives}
        assert "mp-1279" in alt_ids and "mp-1178" in alt_ids  # Fe3O4, FeO
        assert "mp-13" not in alt_ids  # el metal puro nunca aparece

    def test_alternatives_sorted_by_stability(self):
        res = _provider().resolve(
            StructureQuery(elements=("Fe", "O"), qualifier="O")
        )
        ehull = [a.energy_above_hull for a in res.alternatives]
        assert ehull == sorted(ehull)


class TestMpIdFixture:
    def test_mp_id_bypasses_classification(self):
        res = _provider().resolve(StructureQuery(mp_id="mp-1279"))
        assert res.mp_id == "mp-1279"
        assert res.formula == "Fe3O4"  # derivado de la estructura del fixture


class TestSeamIntegration:
    def test_mp_atoms_flow_through_generator(self, tmp_path):
        """R9: los átomos que vienen de MP se escriben por el MISMO pipeline
        (POSCAR/INCAR/KPOINTS) que el camino ASE, sin ramas especiales."""
        res = _provider().resolve(StructureQuery(formula="Fe2O3"))
        gen = VaspInputGenerator(workdir=str(tmp_path), vasp_cmd="vasp_std")
        req = VaspCalcRequest(formula="Fe2O3", calc_kind=CalcKind.STATIC, encut=520)
        result = gen.generate(req, res.atoms)
        run_dir = Path(result.local_dir)
        for name in ("POSCAR", "INCAR", "KPOINTS", "run_vasp.sh"):
            assert (run_dir / name).exists()
        assert set(result.elements) == {"Fe", "O"}
        poscar = (run_dir / "POSCAR").read_text()
        assert "Fe" in poscar and "O" in poscar

    def test_generator_output_depends_only_on_atoms_not_source(self, tmp_path):
        """Mismo `atoms`, misma salida: el generador es agnóstico a la fuente."""
        res = _provider().resolve(StructureQuery(formula="Fe2O3"))
        atoms = res.atoms
        req = VaspCalcRequest(formula="Fe2O3", calc_kind=CalcKind.STATIC, encut=520)
        gen = VaspInputGenerator(workdir=str(tmp_path / "a"), vasp_cmd="vasp_std")
        gen2 = VaspInputGenerator(workdir=str(tmp_path / "b"), vasp_cmd="vasp_std")
        incar1 = (Path(gen.generate(req, atoms.copy()).local_dir) / "INCAR").read_text()
        incar2 = (Path(gen2.generate(req, atoms.copy()).local_dir) / "INCAR").read_text()
        # SYSTEM lleva timestamp; el resto del INCAR debe ser idéntico
        body1 = "\n".join(l for l in incar1.splitlines() if not l.startswith("SYSTEM"))
        body2 = "\n".join(l for l in incar2.splitlines() if not l.startswith("SYSTEM"))
        assert body1 == body2
