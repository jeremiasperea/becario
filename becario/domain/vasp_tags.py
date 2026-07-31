"""Vocabulario de tags del INCAR, tomado del manual de VASP.

Por qué existe: un tag mal escrito NO falla en VASP. Se ignora, la corrida
sale con el default silencioso, y el resultado parece perfectamente válido.
Es la misma clase de error que el NSW=60 o el CONTCAR no convergido: no hay
excepción, hay un número equivocado que nadie mira.

Tener la lista de tags reales convierte ese fallo mudo en uno ruidoso, y es
la precondición para dejar que alguien pida un tag arbitrario desde el bot:
sin vocabulario, aceptar `{"ISMAER": "0"}` significa escribirlo tal cual.

El archivo `vasp_tags.json` lo genera `scripts/build_vasp_tag_vocabulary.py`
desde el manual, y se commitea: generar inputs no puede depender de tener el
manual a mano.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_RUTA = Path(__file__).with_name('vasp_tags.json')

# Tags cuyo valor lo decide BECARIO a partir del TIPO de cálculo, no el
# usuario. NSW, IBRION e ISIF llevan reglas de física ya resueltas (un
# estático no da pasos iónicos; sin pasos iónicos no hay tags iónicos; NSW>1
# sin IBRION es dinámica molecular). Dejarlos overrideables por un pedido
# suelto sería saltearse esas reglas por la puerta de atrás.
TAGS_RESERVADOS = frozenset({'NSW', 'IBRION', 'ISIF', 'SYSTEM'})


@lru_cache(maxsize=1)
def _vocabulario() -> dict[str, dict]:
    return json.loads(_RUTA.read_text(encoding='utf-8'))['tags']


def es_tag_conocido(nombre: str) -> bool:
    """¿`nombre` es un tag del INCAR que el manual documenta?"""
    return nombre.strip().upper() in _vocabulario()


def seccion_de(nombre: str) -> Optional[str]:
    """Sección del manual que documenta el tag (p. ej. `§6.38`), o None.

    Sirve para que los avisos del bot citen de dónde sale la regla en vez de
    afirmarla a secas.
    """
    e = _vocabulario().get(nombre.strip().upper())
    if not e or not e['secs']:
        return None
    return '; '.join(f'§{s}' for s in e['secs'])


def describir(nombre: str) -> str:
    """Descripción corta del tag, o cadena vacía si el manual no la da."""
    e = _vocabulario().get(nombre.strip().upper())
    return e['desc'] if e else ''


def tags_desconocidos(nombres) -> list[str]:
    """Los que NO están en el vocabulario, en orden y sin repetir."""
    vistos, fuera = set(), []
    for n in nombres:
        clave = str(n).strip().upper()
        if clave in vistos:
            continue
        vistos.add(clave)
        if clave not in _vocabulario():
            fuera.append(str(n))
    return fuera


def total_tags() -> int:
    return len(_vocabulario())
