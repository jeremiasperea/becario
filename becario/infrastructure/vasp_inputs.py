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
import time
from pathlib import Path

from ase import Atoms
from ase.io import write

from ..domain.models import CalcDirResult, CalcKind, VaspCalcRequest
from .ase_builder import make_bulk_atoms

logger = logging.getLogger(__name__)

# Densidad de la grilla automática de k-points: n_i = max(1, round(RK / a_i)).
_RK_LENGTH = 30.0  # Å — razonable para metales; el usuario puede pedir otra grilla

# Parámetros comunes a todos los cálculos. ISMEAR=1/SIGMA=0.2 es el smearing
# recomendado para metales (W, Zr…), el caso de uso principal del grupo.
_INCAR_COMMON = {
    "PREC": "Accurate",
    "EDIFF": "1E-6",
    "ISMEAR": 1,
    "SIGMA": 0.2,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}

# La relajación usa ISIF=3: relaja posiciones, forma y volumen de la celda —
# es lo que minimiza los parámetros de red.
_INCAR_BY_KIND = {
    CalcKind.RELAX: {"IBRION": 2, "ISIF": 3, "NSW": 60, "EDIFFG": "-0.01"},
    CalcKind.STATIC: {"NSW": 0},
    CalcKind.ENCUT_SCAN: {"NSW": 0},
}


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
    def generate(self, req: VaspCalcRequest) -> CalcDirResult:
        atoms = make_bulk_atoms(req.formula, req.crystal, req.lattice_a)
        if req.supercell != (1, 1, 1):
            atoms = atoms.repeat(req.supercell)

        kpoints = req.kpoints or _auto_kpoints(atoms)
        encut_values = (
            req.scan_values() if req.calc_kind is CalcKind.ENCUT_SCAN else [req.encut]
        )

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{req.formula}_{req.calc_kind.value}_{stamp}"
        run_dir = self._workdir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        files: list[str] = []
        if req.calc_kind is CalcKind.ENCUT_SCAN:
            for encut in encut_values:
                subdir = run_dir / f"encut_{encut}"
                subdir.mkdir()
                files += self._write_point(subdir, atoms, run_name, kpoints, encut)
        else:
            files += self._write_point(run_dir, atoms, run_name, kpoints, req.encut,
                                       calc_kind=req.calc_kind)

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
        calc_kind: CalcKind = CalcKind.ENCUT_SCAN,
    ) -> list[str]:
        """POSCAR + INCAR + KPOINTS de un punto de cálculo."""
        # sort=True ordena los átomos alfabéticamente por símbolo: el orden
        # de especies del POSCAR queda igual a sorted(set(symbols)), y ese
        # mismo orden usa el servicio para concatenar el POTCAR.
        write(directory / "POSCAR", atoms, format="vasp", direct=True, sort=True)
        (directory / "INCAR").write_text(
            _render_incar(run_name, calc_kind, encut), encoding="utf-8"
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


def _render_incar(run_name: str, calc_kind: CalcKind, encut: int) -> str:
    params: dict = {"SYSTEM": run_name, "ENCUT": encut}
    params.update(_INCAR_COMMON)
    params.update(_INCAR_BY_KIND[calc_kind])
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
