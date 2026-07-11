"""Construcción de estructuras atómicas con ASE (implementa StructureBuilder).

ASE reemplaza al script remoto de la versión n8n: la estructura se arma
localmente, con errores atrapables *antes* de tocar el cluster, y el
archivo resultante (POSCAR/CIF/XYZ) se sube después por SFTP.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from ase import Atoms
from ase.build import bulk, molecule
from ase.io import write

from ..domain.models import (
    OutputFormat,
    StructureKind,
    StructureRequest,
    StructureResult,
)

logger = logging.getLogger(__name__)


class StructureBuildError(RuntimeError):
    """Error de construcción con mensaje apto para mostrar al usuario."""


_EXTENSIONS = {
    OutputFormat.VASP: "POSCAR",  # nombre canónico VASP
    OutputFormat.CIF: "cif",
    OutputFormat.XYZ: "xyz",
}


def make_bulk_atoms(
    formula: str,
    crystal: str | None = None,
    lattice_a: float | None = None,
) -> Atoms:
    """Bulk cristalino vía `ase.build.bulk`, con error apto para el usuario.

    Compartida entre el constructor de estructuras y el generador de inputs
    VASP: ASE conoce la estructura de referencia de los elementos, y para
    compuestos exige red cristalina + parámetro de red.
    """
    kwargs: dict = {}
    if crystal:
        kwargs["crystalstructure"] = crystal
    if lattice_a:
        kwargs["a"] = lattice_a
    try:
        return bulk(formula, **kwargs)
    except Exception as exc:
        raise StructureBuildError(
            f"No pude construir bulk de {formula!r}: {exc}. "
            "Para compuestos indicá red cristalina y parámetro de red "
            "(p. ej. NaCl rocksalt a=5.64)."
        ) from exc


class ASEStructureBuilder:
    def __init__(self, workdir: str = "./structures") -> None:
        self._workdir = Path(workdir)
        self._workdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build(self, req: StructureRequest) -> StructureResult:
        atoms = self._make_atoms(req)

        if req.supercell != (1, 1, 1):
            atoms = atoms.repeat(req.supercell)

        if req.vacuum is not None and req.vacuum > 0:
            atoms.center(vacuum=req.vacuum)

        path = self._write(atoms, req)
        a, b, c = atoms.cell.lengths()
        return StructureResult(
            local_path=str(path),
            filename=path.name,
            chemical_formula=atoms.get_chemical_formula(),
            n_atoms=len(atoms),
            cell_summary=f"a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å",
        )

    # ------------------------------------------------------------------
    def _make_atoms(self, req: StructureRequest) -> Atoms:
        if req.kind is StructureKind.MOLECULE:
            try:
                atoms = molecule(req.formula)
            except KeyError as exc:
                raise StructureBuildError(
                    f"No conozco la molécula {req.formula!r}. "
                    "Probá con una de la base G2 de ASE (H2O, CO2, NH3, CH4…)."
                ) from exc
            # Una molécula sin celda no puede escribirse como POSCAR:
            atoms.center(vacuum=req.vacuum if req.vacuum else 10.0)
            atoms.pbc = True
            return atoms

        return make_bulk_atoms(req.formula, req.crystal, req.lattice_a)

    # ------------------------------------------------------------------
    def _write(self, atoms: Atoms, req: StructureRequest) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if req.output_format is OutputFormat.VASP:
            filename = f"POSCAR_{req.formula}_{stamp}.vasp"
            path = self._workdir / filename
            write(path, atoms, format="vasp", direct=True, sort=True)
        else:
            ext = _EXTENSIONS[req.output_format]
            filename = f"{req.formula}_{stamp}.{ext}"
            path = self._workdir / filename
            write(path, atoms)
        logger.info("Estructura escrita: %s (%d átomos)", path, len(atoms))
        return path
