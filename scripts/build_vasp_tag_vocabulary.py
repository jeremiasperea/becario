#!/usr/bin/env python3
"""Genera el vocabulario de tags del INCAR desde el manual de VASP.

El vocabulario es la lista de tags que BECARIO reconoce como REALES, con la
sección del manual que los documenta. Sirve para que ningún tag llegue a un
INCAR sin haber sido validado: un tag mal escrito no falla en VASP, se ignora
y la corrida sale con el default silencioso, que es peor que un error.

La salida (`becario/domain/vasp_tags.json`) se commitea: el manual no es parte
del repo y no queremos que generar inputs dependa de tenerlo a mano.

Uso:
    .venv/bin/python scripts/build_vasp_tag_vocabulary.py /ruta/al/manual.md

Tres fuentes, en orden de confiabilidad:
  1. la tabla "All parameters" de §6.1 — trae descripción corta y sección;
  2. los encabezados de §6.x, donde el manual marca los tags con backticks;
  3. el cuerpo del texto, para los que solo se documentan en prosa (ADDGRID),
     exigiendo que el tag aparezca ASIGNADO y ADEMÁS entre backticks — pedir
     las dos cosas descarta palabras sueltas en mayúscula y variables Fortran.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TAG_RE = re.compile(r'[A-Z][A-Z0-9_]{1,16}')

# Palabras inglesas, fragmentos y acrónimos que el manual pone entre backticks
# o en negrita dentro de los títulos de §6, y que no son tags del INCAR:
# "DIELECTRIC properties: STATIC and IONIC contributions".
_NO_SON_TAGS = {
    'THE', 'INCAR', 'FILE', 'AND', 'FOR', 'NOT', 'ALL', 'SEC', 'VASP', 'TRUE',
    'FALSE', 'NOTE', 'WARNING', 'DEFAULT', 'ONLY', 'THIS', 'WITH', 'USE',
    'CHI', 'COMPAT', 'CONTR', 'DIELECTRIC', 'EAD', 'ELASTIC', 'END', 'FFT',
    'FROM', 'IMPLICIT', 'IONIC', 'MACROSCOPIC', 'MAG', 'MODULI', 'RELAXATION',
    'STATIC', 'SYMMETRIZED', 'TENSOR', 'TOTAL', 'TS', 'WC', 'XX',
    # Nombres de método (son VALORES de ALGO, no tags) y archivos de salida.
    'GW', 'GW0', 'RPA', 'OUTCAR',
    # Mis-parse del título de §6.38 ("FERDO SMEARINGS tag").
    'SMEARINGS',
    # Bibliotecas, variables de entorno y símbolos del código fuente, no INCAR.
    'BLAS', 'LIB', 'LEXCHG', 'TREAMS',
    # Fragmentos que deja la conversión del PDF al partir un nombre.
    'AMMA', 'OMPAT',
}

# La conversión del PDF perdió los guiones bajos y partió algunos nombres con
# tachados markdown (`GGA` ~~`C`~~ `OMPAT`). El nombre real no es derivable del
# texto dañado, así que estos dos van a mano.
_REPARADOS = {
    'GGA_COMPAT': '§6.71 compatibilidad del modo GGA',
    'LANGEVIN_GAMMA': '§6.62 fricción del termostato de Langevin',
}


def _limpiar(s: str) -> str:
    s = re.sub(r'<br\s*/?>', ' ', s)
    return re.sub(r'[`_*~]', '', s).strip()


def construir(texto: str) -> dict:
    lineas = texto.split('\n')
    tags: dict[str, dict] = {}

    def agregar(nombre: str, desc: str, secciones) -> None:
        nombre = nombre.strip()
        if nombre in _NO_SON_TAGS or not TAG_RE.fullmatch(nombre):
            return
        e = tags.setdefault(nombre, {'desc': '', 'secs': set()})
        if desc and len(desc) > len(e['desc']):
            e['desc'] = desc
        e['secs'].update(secciones)

    # 1) tabla de §6.1: |TAG|descripción (Sec. 6.3,6.11)|
    en_tabla = False
    for ln in lineas:
        if re.match(r'^#{2,4}\s+\**6\.1\b', ln):
            en_tabla = True
            continue
        if en_tabla and re.match(r'^#{2,4}\s+\**6\.2\b', ln):
            break
        if not en_tabla or not ln.startswith('|'):
            continue
        celdas = [_limpiar(c) for c in ln.strip('|').split('|')]
        if len(celdas) < 2 or celdas[0].startswith('---'):
            continue
        secs = re.findall(r'\d+\.\d+', celdas[1])
        corta = re.sub(r'\(Sec\.[^)]*\)', '', celdas[1]).strip()
        for t in re.split(r'[,\s]+', celdas[0]):
            agregar(t, corta, secs)

    # 2) encabezados de §6.x — de 2 a 6 niveles de `#`. Los tags agregados
    # después de que se escribió la tabla de §6.1 (AEXX, NKRED, LMAXMIX) viven
    # justamente en los niveles 5 y 6, que es fácil dejar afuera sin querer.
    encabezado = re.compile(r'^(#{2,6})\s+(.*)$')
    numero = re.compile(r'\**(6(?:\.\d+)*)\**')
    actual = None
    for ln in lineas:
        m = encabezado.match(ln)
        if not m:
            continue
        resto = m.group(2)
        s = numero.match(resto.strip())
        if s:
            actual = s.group(1)
            resto = resto.strip()[s.end():]
        if actual is None:
            continue
        for trozo in re.findall(r'`([^`]+)`', resto):
            for parte in re.split(r'[,\s]+', _limpiar(trozo).split('=')[0]):
                agregar(parte, '', [actual])

    # 3) prosa: asignado Y entre backticks. Las dos condiciones a la vez.
    asignados = set(re.findall(r'\b([A-Z][A-Z0-9_]{2,16})\s*=', texto))
    en_backticks: set[str] = set()
    for trozo in re.findall(r'`([^`\n]{1,50})`', texto):
        for parte in re.split(r'[,\s=]+', _limpiar(trozo)):
            if TAG_RE.fullmatch(parte):
                en_backticks.add(parte)
    for t in asignados & en_backticks:
        agregar(t, '', [])

    for nombre, nota in _REPARADOS.items():
        tags.setdefault(nombre, {'desc': nota, 'secs': set()})

    return {
        k: {'desc': v['desc'], 'secs': sorted(v['secs'])}
        for k, v in sorted(tags.items())
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    manual = Path(sys.argv[1])
    if not manual.is_file():
        print(f'No encuentro el manual en {manual}')
        return 1
    vocab = construir(manual.read_text(encoding='utf-8'))
    destino = Path(__file__).resolve().parent.parent / 'becario' / 'domain' / 'vasp_tags.json'
    destino.write_text(
        json.dumps({'fuente': manual.name, 'tags': vocab}, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    con_desc = sum(1 for v in vocab.values() if v['desc'])
    print(f'{len(vocab)} tags -> {destino}')
    print(f'  con descripción: {con_desc}   con sección: {sum(1 for v in vocab.values() if v["secs"])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
