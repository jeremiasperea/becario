"""Dobles de prueba compartidos (inyectados en tests, sin red ni cluster).

`FakeStructureProvider` permite testear la capa de aplicación (ruteo,
política de nombre ambiguo, mapeo de errores) sin tocar Materials Project.
Vive acá para que PR2 lo reuse cuando implemente el ruteo.
"""
from __future__ import annotations

from ase import Atoms

from becario.domain.models import (
    StructureQuery,
    StructureResolution,
    StructureResolutionError,
)


def demo_atoms() -> Atoms:
    """Un cristal mínimo (NaCl) para no depender de red ni de MP."""
    return Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[5.64, 5.64, 5.64],
        pbc=True,
    )


class FakeStructureProvider:
    """`StructureProvider` de mentira: devuelve una resolución fija o lanza
    un `StructureResolutionError` fijo. Registra cada query recibida para que
    los tests puedan afirmar QUÉ se le pidió a Materials Project."""

    def __init__(
        self,
        resolution: StructureResolution | None = None,
        error: StructureResolutionError | None = None,
    ) -> None:
        self._resolution = resolution
        self._error = error
        self.calls: list[StructureQuery] = []

    def resolve(self, query: StructureQuery) -> StructureResolution:
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        if self._resolution is not None:
            return self._resolution
        return StructureResolution(
            atoms=demo_atoms(),
            mp_id="mp-22862",
            formula="NaCl",
            spacegroup="Fm-3m",
            energy_above_hull=0.0,
            alternatives=(),
        )
