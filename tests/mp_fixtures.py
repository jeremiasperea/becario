"""Carga de fixtures de Materials Project (sintéticos, serializados a JSON).

No son recordings en vivo (MP no es alcanzable en tests): son estructuras
`pymatgen` realistas del sistema Fe-O con mp-ids y `energy_above_hull`
plausibles, para ejercitar selección/filtrado/conversión offline.
"""
from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Structure

_FIXTURE = (
    Path(__file__).parent / "fixtures" / "materials_project" / "fe_o_system.json"
)


class FixtureDoc:
    """Imita un `SummaryDoc` de mp-api con datos de fixture."""

    def __init__(self, data: dict) -> None:
        self.material_id = data["material_id"]
        self.formula_pretty = data["formula_pretty"]
        self.energy_above_hull = data["energy_above_hull"]
        self.structure = Structure.from_dict(data["structure"])


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
