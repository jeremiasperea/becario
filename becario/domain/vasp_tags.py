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

import difflib
import json
import re
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

# Tags que ya tienen un campo dedicado en `VaspCalcRequest`. Aceptarlos
# también por el diccionario genérico crearía dos fuentes de verdad para el
# mismo número, y ganaría la que se escriba último — que es justo el tipo de
# ambigüedad que no se descubre hasta mirar un INCAR raro.
TAGS_CON_CAMPO_PROPIO = frozenset({'ENCUT', 'NBANDS'})

# Un valor va tal cual al INCAR, así que se acota a lo que un valor de VASP
# puede ser: números, .TRUE./.FALSE., palabras como Accurate o Fast, y listas
# separadas por espacios. Lo que importa es lo que NO entra: un salto de línea
# convertiría `{"LREAL": ".FALSE.\nNSW = 999"}` en dos asignaciones y saltearía
# la lista de reservados; un `=` haría lo mismo dentro del renglón.
_VALOR_RE = re.compile(r'^[A-Za-z0-9 ._+\-*/]{1,80}$')


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


def sugerir(nombre: str) -> Optional[str]:
    """El tag más parecido del vocabulario, para un '¿quisiste decir…?'.

    Un typo en un tag es el error MÁS probable de este camino, y el que peor
    falla: VASP no se queja, ignora el tag y corre con el default.
    """
    cerca = difflib.get_close_matches(
        nombre.strip().upper(), list(_vocabulario()), n=1, cutoff=0.75
    )
    return cerca[0] if cerca else None


def validar_tags_pedidos(crudos: dict) -> dict[str, str]:
    """Normaliza y valida tags pedidos a mano. Devuelve `{TAG: valor}`.

    Lanza `ValueError` con un mensaje ya redactado para el usuario. Se rechaza
    en vez de descartar en silencio: un tag ignorado es una corrida que sale
    con otros parámetros que los pedidos y no lo dice.
    """
    limpio: dict[str, str] = {}
    for nombre_crudo, valor_crudo in (crudos or {}).items():
        nombre = str(nombre_crudo).strip().upper()
        valor = str(valor_crudo).strip()

        if nombre in TAGS_RESERVADOS:
            raise ValueError(
                f'{nombre} lo decide el tipo de cálculo, no se pide a mano: '
                'un estático no da pasos iónicos y una relajación sí. '
                'Pedí el tipo de cálculo que querés.'
            )
        if nombre in TAGS_CON_CAMPO_PROPIO:
            raise ValueError(
                f'{nombre} ya tiene su propio campo en el pedido; usá ese en '
                'vez del listado de tags, así no hay dos valores compitiendo.'
            )
        if not es_tag_conocido(nombre):
            parecido = sugerir(nombre)
            extra = f' ¿Quisiste decir {parecido}?' if parecido else ''
            raise ValueError(
                f'{nombre} no es un tag del INCAR que el manual documente.'
                f'{extra} VASP ignoraría un tag mal escrito sin avisar.'
            )
        if not _VALOR_RE.match(valor):
            raise ValueError(
                f'valor inválido para {nombre}: {valor!r}. Se admiten números, '
                '.TRUE./.FALSE., palabras y listas separadas por espacios.'
            )
        limpio[nombre] = valor
    return limpio
