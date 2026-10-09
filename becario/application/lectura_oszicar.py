"""Lectura acotada del OSZICAR, con aviso de truncado.

`read_file` baja solo un PREFIJO del archivo. El último E0 y el conteo de
pasos iónicos viven al FINAL del OSZICAR, así que sobre un prefijo cortado
darían una energía vieja o un conteo de menos. Quien lee un OSZICAR tiene
que saber si llegó entero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..domain.ports import ClusterGateway

# Topes de lectura: el INCAR es diminuto y el OSZICAR crece con los pasos
# iónicos (una línea por paso electrónico).
INCAR_MAX_BYTES = 8_000
OSZICAR_MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class LecturaOszicar:
    # None => el archivo no está (o no se pudo leer).
    texto: Optional[str]
    # True => llegó al tope: lo del final, que es lo que importa, falta.
    truncado: bool


def leer_oszicar(cluster: ClusterGateway, directorio: str) -> LecturaOszicar:
    """Lee `<directorio>/OSZICAR` acotado e informa si quedó cortado."""
    texto = cluster.read_file(f"{directorio}/OSZICAR", max_bytes=OSZICAR_MAX_BYTES)
    return LecturaOszicar(
        texto=texto,
        truncado=texto is not None and len(texto) >= OSZICAR_MAX_BYTES,
    )
