"""Adaptador de Materials Project: implementa el puerto `StructureProvider`.

Aísla TODO el acoplamiento a pymatgen/mp-api. El dominio nunca ve un tipo de
pymatgen: este módulo consulta MP, elige el polimorfo más estable (menor
`energy_above_hull`), lo reduce a la celda primitiva estándar, lo convierte a
`ase.Atoms` y mapea cualquier error crudo a `StructureResolutionError`.
"""
from __future__ import annotations

from typing import Callable, Optional

from mp_api.client import MPRester, MPRestError
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from ..domain.models import (
    StructureAlternative,
    StructureQuery,
    StructureResolution,
    StructureResolutionError,
    StructureResolutionReason,
)

# Campos que pedimos a MP: lo mínimo para elegir y describir el material.
_SUMMARY_FIELDS = [
    "material_id",
    "structure",
    "energy_above_hull",
    "formula_pretty",
]
# Cuántas alternativas ofrecer cuando el nombre era ambiguo.
_MAX_ALTERNATIVES = 5


class MaterialsProjectProvider:
    """`StructureProvider` real. `rester_factory` se inyecta en tests para no
    tocar la red; en producción abre un `MPRester` con la API key."""

    def __init__(
        self,
        api_key: str,
        rester_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self._api_key = api_key
        self._rester_factory = rester_factory or self._default_rester

    def _default_rester(self) -> object:
        return MPRester(api_key=self._api_key)

    # ------------------------------------------------------------------
    def resolve(self, query: StructureQuery) -> StructureResolution:
        try:
            with self._rester_factory() as mpr:
                if query.mp_id:
                    return self._by_mp_id(mpr, query.mp_id)
                if query.formula:
                    docs = self._search(mpr, formula=query.formula)
                else:
                    docs = self._search(mpr, chemsys=_chemsys(query.elements))
                    if query.qualifier:
                        docs = [d for d in docs if _contains(d, query.qualifier)]
                return self._select(docs)
        except StructureResolutionError:
            raise
        except (RequestsConnectionError, RequestsTimeout, ConnectionError, TimeoutError) as exc:
            raise StructureResolutionError(
                StructureResolutionReason.NETWORK, str(exc)
            ) from exc
        except MPRestError as exc:
            raise StructureResolutionError(
                StructureResolutionReason.API, str(exc)
            ) from exc
        except Exception as exc:  # fail-closed: cualquier otro fallo de MP es un ⚠️, no un crash
            raise StructureResolutionError(
                StructureResolutionReason.API, str(exc)
            ) from exc

    # ------------------------------------------------------------------
    def _search(self, mpr, **filters) -> list:
        kwargs = {k: v for k, v in filters.items() if v is not None}
        return list(mpr.materials.summary.search(fields=_SUMMARY_FIELDS, **kwargs))

    def _by_mp_id(self, mpr, mp_id: str) -> StructureResolution:
        structure = mpr.get_structure_by_material_id(mp_id)
        # Con final=True (default) devuelve UNA estructura, pero la API declara
        # `Structure | list[Structure]`: normalizamos para no romper si lista.
        if isinstance(structure, list):
            structure = structure[0] if structure else None
        if structure is None:
            raise StructureResolutionError(StructureResolutionReason.NO_MATCH, mp_id)
        return _to_resolution(
            structure,
            mp_id=mp_id,
            formula=structure.composition.reduced_formula,
            energy_above_hull=0.0,
            alternatives=(),
        )

    def _select(self, docs: list) -> StructureResolution:
        if not docs:
            raise StructureResolutionError(StructureResolutionReason.NO_MATCH)
        # Menor energy_above_hull = más estable; None va al final.
        ordered = sorted(
            docs, key=lambda d: (d.energy_above_hull is None, d.energy_above_hull or 0.0)
        )
        chosen = ordered[0]
        alternatives = tuple(
            StructureAlternative(
                mp_id=str(d.material_id),
                formula=d.formula_pretty,
                energy_above_hull=float(d.energy_above_hull or 0.0),
            )
            for d in ordered[1 : 1 + _MAX_ALTERNATIVES]
        )
        return _to_resolution(
            chosen.structure,
            mp_id=str(chosen.material_id),
            formula=chosen.formula_pretty,
            energy_above_hull=float(chosen.energy_above_hull or 0.0),
            alternatives=alternatives,
        )


# ---------------------------------------------------------------------------
# Helpers puros (conversión pymatgen -> dominio); no tocan la red
# ---------------------------------------------------------------------------


def _chemsys(elements) -> str:
    """Sistema químico en el formato de MP: elementos ordenados, con guiones."""
    return "-".join(sorted(elements))


def _contains(doc, element: str) -> bool:
    """¿La estructura del documento contiene `element`? Filtra el chemsys a los
    óxidos reales (p. ej. descarta Fe puro de una búsqueda 'Fe-O'). Un doc sin
    estructura no se puede verificar: se excluye (no se cuela sin filtrar)."""
    if doc.structure is None:
        return False
    return any(el.symbol == element for el in doc.structure.composition.elements)


def _to_resolution(
    structure,
    *,
    mp_id: str,
    formula: str,
    energy_above_hull: float,
    alternatives: tuple,
) -> StructureResolution:
    sga = SpacegroupAnalyzer(structure)
    primitive = sga.get_primitive_standard_structure()
    atoms = AseAtomsAdaptor.get_atoms(primitive)
    return StructureResolution(
        atoms=atoms,
        mp_id=mp_id,
        formula=formula,
        spacegroup=sga.get_space_group_symbol(),
        energy_above_hull=energy_above_hull,
        alternatives=alternatives,
    )
