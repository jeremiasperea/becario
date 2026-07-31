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
from ase.build import bulk, molecule, surface
from ase.io import write

from ..domain.models import (
    Axis,
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
    conventional: bool = False,
) -> Atoms:
    """Bulk cristalino vía `ase.build.bulk`, con error apto para el usuario.

    Compartida entre el constructor de estructuras y el generador de inputs
    VASP: ASE conoce la estructura de referencia de los elementos, y para
    compuestos exige red cristalina + parámetro de red.

    `conventional=True` pide la celda convencional (`cubic=True`) en vez de la
    primitiva. IMPRESCINDIBLE antes de cortar una superficie: `ase.build.bulk`
    devuelve la PRIMITIVA, y `ase.build.surface` interpreta los índices de
    Miller respecto de esa base. Sobre la primitiva de fluorita, ZrO2 (001) y
    (111) dan la MISMA superficie — pedir una cara y recibir otra, en silencio.
    Las estructuras que no tienen celda cúbica (hcp, tetragonal…) hacen que ASE
    levante `RuntimeError`; ahí la celda que devuelve por default ya ES la
    convencional de esa red, así que se usa tal cual.
    """
    kwargs: dict = {}
    if crystal:
        kwargs["crystalstructure"] = crystal
    if lattice_a:
        kwargs["a"] = lattice_a
    try:
        if conventional:
            try:
                return bulk(formula, cubic=True, **kwargs)
            except RuntimeError:
                logger.debug("Sin celda cúbica para %s; uso la default", formula)
        return bulk(formula, **kwargs)
    except Exception as exc:
        raise StructureBuildError(
            f"No pude construir bulk de {formula!r}: {exc}. "
            "Para compuestos indicá red cristalina y parámetro de red "
            "(p. ej. NaCl rocksalt a=5.64)."
        ) from exc


def make_slab_atoms(
    formula: str,
    crystal: str | None,
    lattice_a: float | None,
    miller: tuple[int, int, int],
    layers: int,
    vacuum: float,
    lattice: Atoms | None = None,
) -> Atoms:
    """Losa con vacío, cortada sobre la celda CONVENCIONAL del cristal.

    `ase.build.surface` deja la normal de la superficie sobre el eje c y el
    vacío repartido a los dos lados, así que la cara que mira al vacío es la
    del índice de Miller pedido.

    `lattice` permite cortar sobre una celda YA RESUELTA (Materials Project,
    o el CONTCAR de una relajación propia) en vez de armarla desde la fórmula.
    OJO: esa celda NO se estandariza, y los índices de Miller se leen en SU
    base. No se estandariza a propósito: sobre una celda relajada el resultado
    depende del `symprec` elegido — para una fluorita con ruido de relajación,
    1e-3 da P1 (celda sin tocar), 1e-2 da R3m (9 átomos, lectura equivocada) y
    5e-2 da Fm-3m (12 átomos, la correcta). Elegir uno por el usuario sería
    volver a meter el error de base que `conventional=True` evita. Quien llama
    tiene que avisar en qué celda se cortó (ver `_resolve_structure`).
    """
    base = (
        lattice
        if lattice is not None
        else make_bulk_atoms(formula, crystal, lattice_a, conventional=True)
    )
    try:
        return surface(base, miller, layers, vacuum=vacuum)
    except Exception as exc:
        hkl = "".join(str(i) for i in miller)
        raise StructureBuildError(
            f"No pude cortar la superficie ({hkl}) de {formula!r}: {exc}."
        ) from exc


def orient_vacuum(atoms: Atoms, axis: Axis) -> Atoms:
    """Acuesta la losa: deja el vacío sobre el eje cartesiano pedido.

    Es una PERMUTACIÓN DE LAS COMPONENTES CARTESIANAS de los vectores de red,
    con las coordenadas fraccionarias intactas — o sea una rotación rígida:
    misma superficie, misma terminación, mismas distancias interatómicas, solo
    cambia qué eje mira al vacío.

    Las dos permutaciones no triviales son CÍCLICAS a propósito. Intercambiar
    dos ejes (lo que parece más directo) es una permutación impar: invierte el
    signo del determinante y deja una celda zurda, que en POSCAR es un volumen
    negativo. Las cíclicas preservan el signo.
    """
    if axis is Axis.Z:  # `surface()` ya deja el vacío sobre c
        return atoms
    # perm[j] = componente vieja que pasa a ocupar la posición j. Se elige de
    # modo que la componente z (donde `surface()` puso el vacío) caiga sobre
    # el eje pedido: y -> [1, 2, 0], x -> [2, 0, 1]. Ambas cíclicas.
    perm = [1, 2, 0] if axis is Axis.Y else [2, 0, 1]
    oriented = atoms.copy()
    scaled = oriented.get_scaled_positions()
    oriented.set_cell(oriented.cell.array[:, perm])
    oriented.set_scaled_positions(scaled)
    return oriented


def describe_slab(req: StructureRequest) -> str | None:
    """Resumen humano de la losa, o None si el pedido no es de losa."""
    if req.kind is not StructureKind.SLAB:
        return None
    hkl = "".join(str(i) for i in req.miller or ())
    return (
        f"cara ({hkl}), {req.layers} capas, {req.vacuum:g} Å de vacío "
        f"en {req.vacuum_axis.value}"
    )


def build_structure_atoms(
    kind: StructureKind,
    formula: str,
    crystal: str | None = None,
    lattice_a: float | None = None,
    miller: tuple[int, int, int] | None = None,
    layers: int | None = None,
    vacuum: float | None = None,
    vacuum_axis: Axis = Axis.Z,
    supercell: tuple[int, int, int] = (1, 1, 1),
    lattice: Atoms | None = None,
) -> Atoms:
    """Bulk o losa, ya con supercelda y orientación. NO escribe archivos.

    El ORDEN de las tres operaciones es lo que importa, y por eso vive en un
    solo lugar: cortar la superficie, repetir la supercelda EN EL PLANO de la
    losa, y recién entonces acostarla. Repetir después de acostar mezclaría los
    índices y un pedido de "2x1 en x,y" dejaría de significar lo que dice.

    Compartida entre `ASEStructureBuilder` y el generador de inputs VASP: los
    dos pedidos replican la parte estructural, así que la construcción tiene
    que ser la misma o pedir una losa por el camino de cálculo devolvería bulk.
    """
    if kind is StructureKind.SLAB:
        # miller/layers/vacuum ya vienen resueltos por `validate_slab_spec`.
        assert miller is not None and layers is not None and vacuum is not None
        atoms = make_slab_atoms(
            formula, crystal, lattice_a, miller, layers, vacuum, lattice=lattice
        )
    elif lattice is not None:
        atoms = lattice.copy()
    else:
        atoms = make_bulk_atoms(formula, crystal, lattice_a)

    if supercell != (1, 1, 1):
        atoms = atoms.repeat(supercell)
    if kind is StructureKind.SLAB:
        atoms = orient_vacuum(atoms, vacuum_axis)
    return atoms


class ASEStructureBuilder:
    def __init__(self, workdir: str = "./structures") -> None:
        self._workdir = Path(workdir)
        self._workdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build(self, req: StructureRequest) -> StructureResult:
        if req.kind is StructureKind.SLAB:
            # La losa ya sale con vacío (`surface()`), supercelda del plano y
            # orientación: NO se le pasa por `center()`, que infla los TRES
            # ejes y convertiría la superficie en un cluster aislado.
            atoms = build_structure_atoms(
                req.kind, req.formula, req.crystal, req.lattice_a,
                miller=req.miller, layers=req.layers, vacuum=req.vacuum,
                vacuum_axis=req.vacuum_axis, supercell=req.supercell,
            )
        else:
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
            slab_summary=describe_slab(req),
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
