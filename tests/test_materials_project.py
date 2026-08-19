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


class _ResterPorIntento:
    """`rester_factory` que entrega un `_FakeRester` distinto por intento y
    cuenta cuántos hubo.

    `_FakeRester` solo, configurado una vez, no puede fallar la primera vez y
    andar la segunda — que es justo el caso que hay que probar. Se lo usa
    como pieza: una lista de resters, uno por intento, y el último se repite
    si el proveedor insiste más veces de las previstas. El contador es la
    forma de ver la política: cuántas veces se llamó de verdad a MP.
    """

    def __init__(self, *resters: _FakeRester):
        self._resters = list(resters)
        self.intentos = 0

    def __call__(self) -> _FakeRester:
        self.intentos += 1
        return self._resters[min(self.intentos, len(self._resters)) - 1]


class TestPoliticaDeReintentoDeMaterialsProject:
    """Consultar Materials Project es seguro de repetir, y por eso se repite.

    La tercera política de reintento del proyecto, y la más permisiva de las
    tres a propósito. Un `sbatch` no se reintenta nunca porque encolar dos
    veces le cuesta horas de cómputo al usuario; una consulta a MP no deja
    nada atrás, así que insistir no puede hacer daño. Esa asimetría es una
    decisión, no un descuido, y estos tests son los que la sostienen.

    Lo que no se reintenta es lo que no va a cambiar por insistir: un
    `NO_MATCH` (el material no está) y un error de la API (MP contestó, y
    contestó mal). Solo se repite `NETWORK`, que es no haber podido hablar.
    """

    def test_un_corte_de_red_pasajero_se_recupera_solo(self):
        """El caso que justifica todo el reintento.

        Con el wifi que se cae un segundo, sin reintento el usuario ve un
        error por algo que ya se arregló para cuando terminó de leerlo. Acá
        el segundo intento contesta y `resolve()` devuelve la estructura sin
        que el fallo llegue nunca a salir del adaptador.
        """
        docs = [_Doc("mp-19770", _iron_oxide(5.0), 0.0, "Fe2O3")]
        factory = _ResterPorIntento(
            _FakeRester(search_error=RequestsConnectionError("se cayó la red")),
            _FakeRester(docs=docs),
        )
        provider = MaterialsProjectProvider(api_key="dummy", rester_factory=factory)

        res = provider.resolve(StructureQuery(formula="Fe2O3"))

        assert isinstance(res, StructureResolution)
        assert res.mp_id == "mp-19770"
        assert factory.intentos == 2, "no reintentó un fallo de red que ya se había ido"

    def test_una_red_que_no_vuelve_se_rinde_a_los_tres_intentos(self):
        """Reintentar es seguro, pero no es gratis: el usuario está esperando.

        Tres intentos y se propaga el error como vino, con reason NETWORK,
        para que la capa de arriba pueda decir 'no pude hablar con MP' en vez
        de 'ese material no existe'.
        """
        factory = _ResterPorIntento(
            _FakeRester(search_error=RequestsConnectionError("sin red"))
        )
        provider = MaterialsProjectProvider(api_key="dummy", rester_factory=factory)

        with pytest.raises(StructureResolutionError) as exc:
            provider.resolve(StructureQuery(formula="Fe2O3"))

        assert exc.value.reason is StructureResolutionReason.NETWORK
        assert factory.intentos == 3

    def test_un_material_que_no_existe_NO_se_reintenta(self):
        """MP contestó, y contestó que no hay nada. Preguntar lo mismo dos
        veces más da la misma respuesta y solo hace esperar al usuario.

        La fórmula es válida a propósito (`Og2O3`: elementos reales, sintaxis
        buena) para que el `NO_MATCH` venga de MP y no del validador de
        `StructureQuery`. Con una fórmula impronunciable el pedido moría antes
        de llegar al adaptador y el test no medía la política de reintento."""
        factory = _ResterPorIntento(_FakeRester(docs=[]))
        provider = MaterialsProjectProvider(api_key="dummy", rester_factory=factory)

        with pytest.raises(StructureResolutionError) as exc:
            provider.resolve(StructureQuery(formula="Og2O3"))

        assert exc.value.reason is StructureResolutionReason.NO_MATCH
        assert factory.intentos == 1, "reintentó una búsqueda sin resultados"

    def test_un_error_de_la_api_NO_se_reintenta(self):
        """Una API key vencida o una consulta mal formada no se arreglan
        insistiendo: el servidor está vivo y ya dijo que no."""
        factory = _ResterPorIntento(_FakeRester(search_error=MPRestError("boom")))
        provider = MaterialsProjectProvider(api_key="dummy", rester_factory=factory)

        with pytest.raises(StructureResolutionError) as exc:
            provider.resolve(StructureQuery(formula="Fe2O3"))

        assert exc.value.reason is StructureResolutionReason.API
        assert factory.intentos == 1, "reintentó un error que MP ya había contestado"
