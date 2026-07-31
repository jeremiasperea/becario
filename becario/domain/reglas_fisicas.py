"""Avisos sobre combinaciones de parámetros que el manual desaconseja.

La diferencia con `vasp_tags`: allá se valida que un tag EXISTA; acá se mira
si la combinación tiene sentido. `ISMEAR=-5` es un valor perfectamente
válido, y usarlo para relajar un metal da fuerzas malas — el INCAR se genera
igual, VASP corre igual, y el resultado es basura silenciosa.

Estos avisos NO bloquean. Igual que con el CONTCAR no convergido, la decisión
es de quien hace la física: el bot pone el dato sobre la mesa y quien pidió
el cálculo decide. Cada aviso cita la sección del manual de donde sale, así
la afirmación es verificable y no hay que creerle al bot.

Las reglas salen del texto de §6.38, que resume:

  - "For relaxations *in metals* always use ISMEAR=1 or ISMEAR=2 and an
    appropriate SIGMA value"
  - "For semiconductors or insulators use the tetrahedron method
    (ISMEAR=-5), if the cell is too large (or if you use only a single
    k-point) use ISMEAR=0"
  - "For the calculations of the DOS and very accurate total energy
    calculations (no relaxation in metals) use the tetrahedron method"
  - los métodos de tetraedros (-4, -5) piden una malla centrada en Γ
"""
from __future__ import annotations

from typing import Optional

from .models import CalcKind

# Métodos de tetraedros: no llevan ancho de smearing y necesitan malla Γ.
_TETRAEDROS = {'-4', '-5'}


def _valor(tags: dict, nombre: str) -> Optional[str]:
    v = (tags or {}).get(nombre)
    return str(v).strip() if v is not None else None


def advertencias(
    calc_kind: CalcKind,
    incar_tags: Optional[dict] = None,
    kpoints: Optional[tuple[int, int, int]] = None,
) -> list[str]:
    """Avisos para esta combinación. Lista vacía si no hay nada que decir.

    `kpoints` es la grilla ya resuelta; si no se pasa, las reglas que
    dependen del muestreo se omiten en vez de adivinar.
    """
    tags = incar_tags or {}
    ismear = _valor(tags, 'ISMEAR')
    avisos: list[str] = []

    if ismear in _TETRAEDROS and calc_kind is CalcKind.RELAX:
        avisos.append(
            f'⚠️ ISMEAR={ismear} (tetraedros) para relajar: el manual pide '
            'ISMEAR=1 o 2 con un SIGMA apropiado para relajaciones en metales. '
            'Con tetraedros las fuerzas salen mal y la relajación converge a '
            'una geometría equivocada, sin que VASP se queje (§6.38).'
        )

    if ismear in _TETRAEDROS and _valor(tags, 'SIGMA') is not None:
        avisos.append(
            f'ℹ️ Con ISMEAR={ismear} el SIGMA que pediste no se usa: los '
            'métodos de tetraedros no llevan ancho de smearing (§6.38).'
        )

    if ismear in _TETRAEDROS and kpoints is not None and all(k == 1 for k in kpoints):
        avisos.append(
            f'⚠️ ISMEAR={ismear} con un solo punto k: el método de tetraedros '
            'necesita una malla. Para celdas grandes o un único punto k el '
            'manual recomienda ISMEAR=0 (§6.38).'
        )

    if ismear == '0' and _valor(tags, 'SIGMA') is None:
        avisos.append(
            'ℹ️ ISMEAR=0 (gaussiano) sin SIGMA explícito: queda el 0.2 del '
            'template, que es un valor pensado para metales. Para aislantes '
            'suele quererse bastante más chico (§6.38).'
        )

    return avisos
