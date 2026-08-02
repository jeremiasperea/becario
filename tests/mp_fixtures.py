"""Carga de fixtures de Materials Project (sintéticos, serializados a JSON).

No son recordings en vivo (MP no es alcanzable en tests): son estructuras
`pymatgen` realistas del sistema Fe-O con mp-ids y `energy_above_hull`
plausibles, para ejercitar selección/filtrado/conversión offline.
"""
from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

_FIXTURE = (
    Path(__file__).parent / "fixtures" / "materials_project" / "fe_o_system.json"
)


class _Symmetry:
    """La parte de `SymmetryData` que el adaptador mira."""

    def __init__(self, crystal_system: str) -> None:
        self.crystal_system = crystal_system


class FixtureDoc:
    """Imita un `SummaryDoc` de mp-api con datos de fixture."""

    def __init__(self, data: dict) -> None:
        self.material_id = data["material_id"]
        self.formula_pretty = data["formula_pretty"]
        self.energy_above_hull = data["energy_above_hull"]
        self.structure = Structure.from_dict(data["structure"])
        # El sistema cristalino se DERIVA de la estructura del fixture en vez
        # de guardarse en el JSON: así no puede quedar desincronizado de los
        # átomos que lo acompañan.
        self.symmetry = _Symmetry(
            SpacegroupAnalyzer(self.structure).get_crystal_system().capitalize()
        )


class PolymorphDoc:
    """Doc con la fase declarada a mano, para ejercitar la SELECCIÓN entre
    polimorfos.

    OJO: `structure` es de relleno —se reusa una del fixture Fe-O— y NO se
    corresponde con `crystal_system`. Estos docs sirven para probar el
    filtrado y el orden por estabilidad, no la física de ninguna fase. Un
    fixture con las tres fases reales de ZrO2 haría falta recién para probar
    la construcción, que acá no se toca.
    """

    def __init__(
        self,
        mp_id: str,
        formula: str,
        energy_above_hull: float,
        crystal_system: str,
        structure: Structure,
    ) -> None:
        self.material_id = mp_id
        self.formula_pretty = formula
        self.energy_above_hull = energy_above_hull
        self.symmetry = _Symmetry(crystal_system)
        self.structure = structure


def zro2_polymorph_docs() -> list[PolymorphDoc]:
    """Las tres fases de ZrO2 por estabilidad: monoclínica (la de ambiente,
    en el hull), tetragonal y cúbica. Los `energy_above_hull` reflejan ese
    orden; las estructuras son de relleno (ver `PolymorphDoc`)."""
    filler = fe_o_docs()[0].structure
    return [
        PolymorphDoc("mp-2858", "ZrO2", 0.0, "Monoclinic", filler),
        PolymorphDoc("mp-1565", "ZrO2", 0.045, "Tetragonal", filler),
        PolymorphDoc("mp-1018721", "ZrO2", 0.089, "Cubic", filler),
    ]


def fe_o_docs() -> list[FixtureDoc]:
    return [FixtureDoc(d) for d in json.loads(_FIXTURE.read_text(encoding="utf-8"))]


class FixtureRester:
    """`MPRester` de mentira alimentado por los fixtures: filtra por fórmula
    igual que MP; para chemsys devuelve todo el sistema (el filtro por
    elemento cualificador lo hace el adaptador)."""

    def __init__(self, docs: list[FixtureDoc]) -> None:
        self._docs = docs

    def __enter__(self) -> "FixtureRester":
        return self

    def __exit__(self, *a) -> bool:
        return False

    @property
    def materials(self) -> "FixtureRester":
        return self

    @property
    def summary(self) -> "FixtureRester":
        return self

    def search(self, formula=None, chemsys=None, fields=None) -> list[FixtureDoc]:
        docs = self._docs
        if formula is not None:
            docs = [d for d in docs if d.formula_pretty == formula]
        return list(docs)

    def get_structure_by_material_id(self, mp_id: str, **kw):
        for d in self._docs:
            if d.material_id == mp_id:
                return d.structure
        return None
