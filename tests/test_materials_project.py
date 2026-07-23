"""Tests del adaptador MaterialsProjectProvider (tareas 1.4-1.8 de PR1).

El `MPRester` se mockea (sin red), pero las estructuras son `Structure` reales
de pymatgen para que la reducción a primitiva + conversión a `ase.Atoms` se
ejerciten de verdad. Cubre R2 (formula), R3 (chemsys+filtro+alternativas),
R4 (mp-id), R9 (conversión) y R6/R7 (mapeo de errores).
"""
from __future__ import annotations

import pytest
from ase import Atoms
from mp_api.client import MPRestError
from pymatgen.core import Lattice, Structure
from requests.exceptions import ConnectionError as RequestsConnectionError

from becario.domain.models import (
    StructureQuery,
    StructureResolution,
    StructureResolutionError,
    StructureResolutionReason,
)
from becario.infrastructure.materials_project import MaterialsProjectProvider


# --- estructuras reales mínimas -------------------------------------------


def _fe_metal() -> Structure:
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _iron_oxide(a: float) -> Structure:
    return Structure(
        Lattice.cubic(a),
        ["Fe", "Fe", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0.5]],
    )


class _Doc:
    def __init__(self, material_id, structure, energy_above_hull, formula_pretty):
        self.material_id = material_id
        self.structure = structure
        self.energy_above_hull = energy_above_hull
        self.formula_pretty = formula_pretty


class _FakeRester:
    """Doble de MPRester: context manager con `materials.summary.search` y
    `get_structure_by_material_id`, configurable para devolver o fallar."""

    def __init__(self, *, docs=None, structure=None, search_error=None):
        self._docs = docs
        self._structure = structure
        self._search_error = search_error
        self.search_kwargs = None
        # mpr.materials.summary.search -> self.search
        self.materials = self
        self.summary = self

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        if self._search_error is not None:
            raise self._search_error
        return list(self._docs or [])

    def get_structure_by_material_id(self, mp_id, **kw):
        return self._structure

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _provider(fake: _FakeRester) -> MaterialsProjectProvider:
    return MaterialsProjectProvider(api_key="dummy", rester_factory=lambda: fake)


class TestMpIdBranch:
    def test_resolves_by_material_id(self):
        fake = _FakeRester(structure=_iron_oxide(5.0))
        res = _provider(fake).resolve(StructureQuery(mp_id="mp-19770"))
        assert isinstance(res, StructureResolution)
        assert res.mp_id == "mp-19770"
        assert isinstance(res.atoms, Atoms)
        assert "Fe" in res.atoms.get_chemical_symbols()

    def test_missing_structure_is_no_match(self):
        fake = _FakeRester(structure=None)
        with pytest.raises(StructureResolutionError) as exc:
            _provider(fake).resolve(StructureQuery(mp_id="mp-1"))
        assert exc.value.reason is StructureResolutionReason.NO_MATCH

    def test_list_return_is_normalized(self):
        # get_structure_by_material_id puede devolver Structure | list[Structure]
        fake = _FakeRester(structure=[_iron_oxide(5.0)])
        res = _provider(fake).resolve(StructureQuery(mp_id="mp-19770"))
        assert res.mp_id == "mp-19770"
        assert "Fe" in res.atoms.get_chemical_symbols()

    def test_empty_list_return_is_no_match(self):
        fake = _FakeRester(structure=[])
        with pytest.raises(StructureResolutionError) as exc:
            _provider(fake).resolve(StructureQuery(mp_id="mp-1"))
        assert exc.value.reason is StructureResolutionReason.NO_MATCH


class TestFormulaBranch:
    def test_picks_lowest_energy_above_hull(self):
        docs = [
            _Doc("mp-2", _iron_oxide(5.2), 0.05, "Fe2O3"),
            _Doc("mp-1", _iron_oxide(5.0), 0.00, "Fe2O3"),
        ]
        fake = _FakeRester(docs=docs)
        res = _provider(fake).resolve(StructureQuery(formula="Fe2O3"))
        assert res.mp_id == "mp-1"
        assert fake.search_kwargs.get("formula") == "Fe2O3"
        # el otro candidato queda como alternativa
        assert res.alternatives[0].mp_id == "mp-2"


class TestChemsysBranch:
    def test_filters_by_qualifier_and_returns_alternatives(self):
        docs = [
            _Doc("mp-fe", _fe_metal(), 0.00, "Fe"),          # metal puro: se descarta
            _Doc("mp-hem", _iron_oxide(5.0), 0.00, "Fe2O3"),  # óxido más estable
            _Doc("mp-mag", _iron_oxide(5.3), 0.03, "Fe3O4"),  # óxido alternativo
        ]
        fake = _FakeRester(docs=docs)
        res = _provider(fake).resolve(
            StructureQuery(elements=("Fe", "O"), qualifier="O")
        )
        assert fake.search_kwargs.get("chemsys") == "Fe-O"
        assert res.mp_id == "mp-hem"  # Fe puro filtrado, óxido más estable elegido
        alt_ids = [a.mp_id for a in res.alternatives]
        assert "mp-mag" in alt_ids
        assert "mp-fe" not in alt_ids

    def test_no_results_is_no_match(self):
        fake = _FakeRester(docs=[])
        with pytest.raises(StructureResolutionError) as exc:
            _provider(fake).resolve(StructureQuery(elements=("Fe", "O")))
        assert exc.value.reason is StructureResolutionReason.NO_MATCH


class TestConversion:
    def test_returns_primitive_ase_atoms_with_spacegroup(self):
        docs = [_Doc("mp-1", _iron_oxide(5.0), 0.0, "Fe2O3")]
        res = _provider(_FakeRester(docs=docs)).resolve(StructureQuery(formula="Fe2O3"))
        assert isinstance(res.atoms, Atoms)
        assert len(res.atoms) >= 1
        assert res.spacegroup  # símbolo de grupo espacial no vacío


class TestErrorMapping:
    def test_api_error_maps_to_api_reason(self):
        fake = _FakeRester(search_error=MPRestError("boom"))
        with pytest.raises(StructureResolutionError) as exc:
            _provider(fake).resolve(StructureQuery(formula="Fe2O3"))
        assert exc.value.reason is StructureResolutionReason.API

    def test_connection_error_maps_to_network_reason(self):
        fake = _FakeRester(search_error=RequestsConnectionError("no net"))
        with pytest.raises(StructureResolutionError) as exc:
            _provider(fake).resolve(StructureQuery(formula="Fe2O3"))
        assert exc.value.reason is StructureResolutionReason.NETWORK
