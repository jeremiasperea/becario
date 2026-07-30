"""Generación local de inputs VASP completos (implementa CalcInputGenerator).

Mismo principio que `ase_builder`: todo se arma localmente, con errores
atrapables *antes* de tocar el cluster, y el directorio resultante se sube
después por SFTP. El POTCAR es la única pieza que NO se genera acá: se
concatena en el cluster desde la biblioteca de pseudopotenciales
(`BECARIO_POTCAR_DIR`), porque los POTCAR son licenciados y no viven en
esta máquina.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

from ase import Atoms
from ase.io import write

from ..domain.models import CalcDirResult, CalcKind, VaspCalcRequest
from .ase_builder import build_structure_atoms

logger = logging.getLogger(__name__)

# Densidad de la grilla automática de k-points: n_i = max(1, round(RK / a_i)).
_RK_LENGTH = 30.0  # Å — razonable para metales; el usuario puede pedir otra grilla

# Parámetros comunes a todos los cálculos. Base tomada de un INCAR de producción
# del grupo (interfaz ZrO2-Zr), quedándonos solo con los tags de CALIDAD que
# valen para cualquier sistema; los específicos del sistema de origen (NBANDS,
# ISMEAR=-5 de tetraedros, mixing agresivo, DOS, U) NO se copian.
#
# ISMEAR=1/SIGMA=0.2 (Methfessel-Paxton) es el smearing correcto para relajar
# metales (W, Zr…), el caso de uso principal del grupo — NO tetraedros (-5),
# que da fuerzas malas al relajar.
#
# LREAL=.FALSE. es una desviación deliberada respecto del INCAR de referencia
# (que usa `auto`): la referencia es un slab de ~42 átomos, pero acá se arman
# celdas chicas (bulk de pocos átomos), donde VASP recomienda proyección en
# espacio recíproco (.FALSE.), no real.
_INCAR_COMMON = {
    "PREC": "Accurate",
    "EDIFF": "1E-6",
    "ISMEAR": 1,
    "SIGMA": 0.2,
    "ALGO": "Fast",
    "LREAL": ".FALSE.",
    "NELMIN": 10,
    "ADDGRID": ".TRUE.",
    "LASPH": ".TRUE.",
    "LMAXMIX": 6,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}

# ISIF por tipo de cálculo cuando el pedido no lo fuerza: 3 relaja posiciones,
# forma y volumen — es lo que minimiza los parámetros de red. Solo se emite si
# hay pasos iónicos (ver `_render_incar`).
_DEFAULT_ISIF = {CalcKind.RELAX: 3}

# Presupuesto de pasos iónicos por tipo de cálculo. TODOS los tipos tienen
# entrada explícita y el NSW se emite SIEMPRE, sin dejarle el default a VASP:
# para NSW>1 sin IBRION, VASP asume IBRION=0, que es dinámica molecular.
#
# 180 es el valor de producción del grupo. El 60 que estuvo acá hasta ahora
# venía del andamio inicial (commit b34991a) y sobrevivió a la migración al
# template de producción (6cf1906) porque esa revisión auditó smearing, NBANDS
# e ISIF pero no el presupuesto iónico. Un presupuesto corto no falla ruidoso:
# la corrida termina con código 0 y deja un CONTCAR a medio relajar que parece
# perfectamente válido.
_DEFAULT_NSW = {
    CalcKind.RELAX: 180,
    CalcKind.STATIC: 0,
    CalcKind.ENCUT_SCAN: 0,
}

# Tags que solo tienen sentido si hay pasos iónicos. Se emiten según el NSW
# RESUELTO, no según el tipo de cálculo: así "sin pasos iónicos no hay tags
# iónicos" queda dicho una sola vez y sigue valiendo para los tipos que se
# agreguen después (bandas, DOS).
#
# EDIFFG=-0.02 eV/Å es el criterio de fuerzas del grupo. El -0.01 que estuvo
# acá tenía el mismo origen que el NSW=60: andamio inicial, nunca auditado
# contra la práctica real. No se expone como override porque es un valor que
# casi nunca se cambia; si algún día hace falta, sigue el molde de `isif`.
_IONIC_TAGS = {"IBRION": 2, "EDIFFG": "-0.02"}

# Electrones de valencia (ZVAL) por elemento, para estimar NBANDS por sistema.
# CLAVE: cada valor corresponde al POTCAR EXACTO que el cluster concatena para
# ese elemento (ver `_POTCAR_VARIANTS` en services: prueba `_sv`, `_pv`, `""`
# y usa el primero que existe). Estos valores se leyeron de POTCAR reales
# (potpaw_PBE), NO de memoria. Si en el cluster cambia la variante disponible
# para un elemento, hay que actualizar su ZVAL acá. Un elemento ausente =>
# NBANDS lo decide VASP.
_ZVAL: dict[str, int] = {
    "Zr": 12,  # resuelve a Zr_sv
    "O": 6,    # resuelve a O
    "W": 12,   # sin W_sv en potpaw_PBE => resuelve a W_pv
}

# Margen de bandas vacías sobre el mínimo ocupado (NELECT/2). Holgado a
# propósito: si la tabla ZVAL subestima el conteo, VASP 5.4.4 igual auto-sube
# NBANDS con un warning en vez de romper, así que preferimos pasarnos.
_NBANDS_MARGIN = 1.2


class VaspInputGenerator:
    def __init__(
        self,
        workdir: str = "./structures",
        vasp_cmd: str = "vasp_std",
        vasp_prelude: str = "",
    ) -> None:
        self._workdir = Path(workdir) / "runs"
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._vasp_cmd = vasp_cmd
        self._vasp_prelude = vasp_prelude.strip()

    # ------------------------------------------------------------------
    def generate(self, req: VaspCalcRequest, atoms: Atoms | None = None) -> CalcDirResult:
        # `atoms` ya resuelto (p. ej. desde Materials Project) se usa tal cual;
        # None => se arma con ASE desde la fórmula/red del pedido. El generador
        # es agnóstico a la FUENTE de la estructura: solo la escribe. La
        # supercelda y todo lo de abajo es idéntico sea cual sea el origen.
        # Una estructura ya resuelta (Materials Project, o el CONTCAR de una
        # relajación propia) entra como `lattice`: se usa tal cual para un
        # bulk, y como celda de partida para CORTAR la losa si el pedido es de
        # superficie. Antes se devolvía la celda entera sin cortar — un bulk
        # con nombre de slab, sin que nada avisara.
        atoms = build_structure_atoms(
            req.kind, req.formula, req.crystal, req.lattice_a,
            miller=req.miller, layers=req.layers, vacuum=req.vacuum,
            vacuum_axis=req.vacuum_axis, supercell=req.supercell,
            lattice=atoms,
        )

        kpoints = req.kpoints or _auto_kpoints(atoms)
        encut_values = (
            req.scan_values() if req.calc_kind is CalcKind.ENCUT_SCAN else [req.encut]
        )

        # NBANDS e ISIF son iguales en todos los puntos (misma estructura y
        # tipo de cálculo), así que se resuelven una sola vez acá.
        nbands = req.nbands if req.nbands is not None else _auto_nbands(atoms)
        nsw, nsw_forced = _resolve_nsw(req.calc_kind, req.nsw)
        if nsw_forced:
            # No se corrige en silencio: queda registrado que lo pedido no es
            # lo que se emitió. Cuando el NSW sea pedible desde el bot, este
            # mismo flag es el que tiene que llegar a la respuesta del usuario.
            logger.warning(
                "NSW=%s ignorado: %s es un cálculo de un solo punto, se fuerza NSW=0",
                req.nsw, req.calc_kind.value,
            )
        # ISIF solo tiene efecto con NSW>0, así que se resuelve (default o
        # override del pedido) únicamente cuando hay pasos iónicos.
        isif = None
        if nsw > 0:
            isif = req.isif if req.isif is not None else _DEFAULT_ISIF.get(req.calc_kind)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{req.formula}_{req.calc_kind.value}_{stamp}"
        run_dir = self._workdir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        files: list[str] = []
        if req.calc_kind is CalcKind.ENCUT_SCAN:
            for encut in encut_values:
                subdir = run_dir / f"encut_{encut}"
                subdir.mkdir()
                files += self._write_point(
                    subdir, atoms, run_name, kpoints, encut, nsw=nsw, nbands=nbands
                )
        else:
            files += self._write_point(
                run_dir, atoms, run_name, kpoints, req.encut,
                nsw=nsw, nbands=nbands, isif=isif,
            )

        script = run_dir / "run_vasp.sh"
        script.write_text(self._job_script(req.calc_kind), encoding="utf-8")
        files.append("run_vasp.sh")

        a, b, c = atoms.cell.lengths()
        result = CalcDirResult(
            local_dir=str(run_dir),
            run_name=run_name,
            files=[str(Path(f).as_posix()) for f in files],
            elements=sorted(set(atoms.get_chemical_symbols())),
            chemical_formula=atoms.get_chemical_formula(),
            n_atoms=len(atoms),
            cell_summary=f"a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å",
            kpoints=kpoints,
            calc_kind=req.calc_kind,
            encut_values=encut_values,
        )
        logger.info(
            "Inputs VASP generados: %s (%s, %d archivos)",
            run_dir, req.calc_kind.value, len(result.files),
        )
        return result

    # ------------------------------------------------------------------
    def _write_point(
        self,
        directory: Path,
        atoms: Atoms,
        run_name: str,
        kpoints: tuple[int, int, int],
        encut: int,
        nsw: int = 0,
        nbands: int | None = None,
        isif: int | None = None,
    ) -> list[str]:
        """POSCAR + INCAR + KPOINTS de un punto de cálculo."""
        # sort=True ordena los átomos alfabéticamente por símbolo: el orden
        # de especies del POSCAR queda igual a sorted(set(symbols)), y ese
        # mismo orden usa el servicio para concatenar el POTCAR.
        write(directory / "POSCAR", atoms, format="vasp", direct=True, sort=True)
        (directory / "INCAR").write_text(
            _render_incar(run_name, encut, nsw, nbands=nbands, isif=isif),
            encoding="utf-8",
        )
        (directory / "KPOINTS").write_text(_render_kpoints(kpoints), encoding="utf-8")
        rel = directory.name + "/" if directory.name.startswith("encut_") else ""
        return [f"{rel}POSCAR", f"{rel}INCAR", f"{rel}KPOINTS"]

    def _job_script(self, calc_kind: CalcKind) -> str:
        """Script de corrida. Hace `cd` a su propio directorio: el monitor
        asume que el directorio del script es el de la corrida."""
        lines = ['#!/bin/bash', 'set -e', 'cd "$(dirname "$0")"']
        if self._vasp_prelude:
            lines.append(self._vasp_prelude)
        if calc_kind is CalcKind.ENCUT_SCAN:
            # El POTCAR vive en la raíz de la corrida (se arma una sola vez
            # en el cluster) y se copia a cada punto del barrido.
            lines += [
                "for d in encut_*/; do",
                '  cp POTCAR "$d"',
                f'  ( cd "$d" && {self._vasp_cmd} > vasp.out 2>&1 )',
                "done",
            ]
        else:
            lines.append(f"{self._vasp_cmd} > vasp.out 2>&1")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Renderizado de archivos VASP
# ---------------------------------------------------------------------------


def _auto_kpoints(atoms: Atoms) -> tuple[int, int, int]:
    a, b, c = atoms.cell.lengths()
    return tuple(max(1, round(_RK_LENGTH / length)) for length in (a, b, c))


def _auto_nbands(atoms: Atoms) -> int | None:
    """Estima NBANDS a partir del conteo de electrones de valencia.

    Devuelve None si algún elemento no está en la tabla ZVAL: en ese caso el
    INCAR se emite sin NBANDS y VASP lo calcula con el ZVAL real del POTCAR.
    """
    symbols = atoms.get_chemical_symbols()
    if any(sym not in _ZVAL for sym in symbols):
        return None
    nelect = sum(_ZVAL[sym] for sym in symbols)
    return math.ceil((nelect / 2) * _NBANDS_MARGIN)


def _resolve_nsw(calc_kind: CalcKind, requested: int | None) -> tuple[int, bool]:
    """NSW efectivo, y si hubo que forzarlo contra lo que pedía el request.

    El TIPO DE CÁLCULO manda: los tipos de un solo punto llevan NSW=0 por
    definición, así que un pedido con pasos iónicos se fuerza a 0 — pero
    devolviendo el flag, para que quede dicho y no sea una corrección muda.

    Forzar acá en vez de rechazar además vuelve INALCANZABLE la trampa del
    IBRION: como el único tipo que admite NSW>0 es el que relaja, y ese sí
    emite IBRION=2, nunca se puede llegar a un INCAR con NSW>1 sin IBRION
    (que VASP interpretaría como dinámica molecular).
    """
    default = _DEFAULT_NSW[calc_kind]
    if default == 0:
        return 0, requested not in (None, 0)
    return (default if requested is None else requested), False


def _render_incar(
    run_name: str,
    encut: int,
    nsw: int,
    nbands: int | None = None,
    isif: int | None = None,
) -> str:
    params: dict = {"SYSTEM": run_name, "ENCUT": encut}
    params.update(_INCAR_COMMON)
    params["NSW"] = nsw
    if nsw > 0:
        params.update(_IONIC_TAGS)
        if isif is not None:
            params["ISIF"] = isif
    if nbands is not None:
        params["NBANDS"] = nbands
    return "\n".join(f"{key} = {value}" for key, value in params.items()) + "\n"


def _render_kpoints(kpoints: tuple[int, int, int]) -> str:
    kx, ky, kz = kpoints
    return (
        "KPOINTS generada por BECARIO\n"
        "0\n"
        "Gamma\n"
        f"{kx} {ky} {kz}\n"
        "0 0 0\n"
    )
