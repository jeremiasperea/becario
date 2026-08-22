"""Datos puntuales de una corrida VASP, y cómo sacarlos de sus archivos.

Nace de medir E6 (`docs/plan_lo_que_mostro_usarlo.md`). El caso real:

    vos> Muéstrame el OSZICAR
    bot> 📄 …/OSZICAR: [el archivo, truncado]
    vos> Cuantas vueltas iónicas hizo?
    bot> ❓ No pude interpretar tu pedido.

El plan lo anotaba como el defecto más grande de los seis, porque «requiere
que el bot interprete un archivo, no que lo muestre». La medición dijo otra
cosa: **no pide interpretación, pide un parseo**, y seis de nueve de esos
parseos ya estaban escritos en el repo. El bot cuenta los pasos iónicos de
cada relajación —en `relaxed_source._check_convergence`, para avisar si se
quedó sin NSW— y cuando le preguntaron el número contestó que no entendía.

Lo que faltaba era poder DECIRLO: el ruteo emitía el mismo plan byte a byte
para «cuántas vueltas iónicas hizo» y «qué energía dio» (3 planes para 9
hechos distintos), así que el pedido se perdía antes de llegar al handler.

Este módulo es el vocabulario de ese eje, del lado del dominio. Vive acá y
no en el router por la misma razón que `ASE_CRYSTALS`: es lo que el sistema
sabe contestar, no un detalle del modelo que lo interpreta. El router
importa el enum para armar su schema, así que las dos listas no pueden
desincronizarse.

Solo entra lo que se sabe calcular. Es la misma decisión que el rebote de
las heterostructuras y que el barrido de k-points en las sugerencias: **no
ofrecer lo que no se sabe hacer.** El diagnóstico de un fallo, por ejemplo,
queda AFUERA aunque el bot lo arme: hoy vive en el monitor, atado al aviso
proactivo, y traerlo hasta acá es su propio trabajo. Ponerlo en la lista
antes de eso sería prometer algo que este camino no cumple.

Las funciones son puras: reciben el TEXTO del archivo, no el cluster. Eso
las deja del lado del dominio, testeables sin red, y es lo que permite
verificarlas contra un fragmento real en
`scripts/medir_preguntas_de_contenido.py`.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

# Un paso iónico por línea con "F=" en el OSZICAR. La misma expresión que
# usa `relaxed_source` para contar contra NSW: es literalmente el número que
# el usuario pidió, calculado ya para otra cosa.
_PASO_IONICO_RE = re.compile(r"^\s*\d+\s+F=", re.MULTILINE)
_NSW_RE = re.compile(r"^\s*NSW\s*=\s*(\d+)", re.MULTILINE | re.IGNORECASE)
_E0_RE = re.compile(r"E0=\s*([-+0-9.Ee]+)")


class DatoDeCorrida(str, Enum):
    """Qué dato puntual pide una pregunta sobre una corrida ya terminada.

    Medido con `qwen2.5-coder:14b`: sobre nueve preguntas, el modelo eligió
    bien las que están acá y se abstuvo en las que no, 3/3 unánime. Ese
    último número es el que importa — un vocabulario que nunca se abstiene
    contesta la energía cuando le piden un resumen.
    """

    PASOS_IONICOS = "pasos_ionicos"
    ENERGIA = "energia"
    CONVERGENCIA = "convergencia"
    ATOMOS = "atomos"
    PARAMETROS_RED = "parametros_red"


# Lo que el modelo contesta cuando la pregunta cae FUERA del vocabulario.
# No es un valor del enum a propósito: es la ausencia de dato, y tratarlo
# como uno más invita a que algún camino lo intente calcular.
NINGUNO = "ninguno"

# Cómo se le describe cada dato al modelo. Vive junto al enum para que
# agregar un valor y olvidarse de explicarlo sea imposible de pasar por
# alto: el prompt del router se arma con esto.
DESCRIPCIONES: dict[DatoDeCorrida, str] = {
    DatoDeCorrida.PASOS_IONICOS: "cuántas vueltas o pasos iónicos hizo la relajación",
    DatoDeCorrida.ENERGIA: "la energía final del cálculo",
    DatoDeCorrida.CONVERGENCIA: "si la relajación llegó al criterio o se quedó sin pasos",
    DatoDeCorrida.ATOMOS: "cuántos átomos tiene la celda",
    DatoDeCorrida.PARAMETROS_RED: "a, b, c y los ángulos de la celda",
}


def dato_valido(crudo: Optional[str]) -> Optional[DatoDeCorrida]:
    """El `DatoDeCorrida` que nombra `crudo`, o `None`.

    `None` cubre los tres casos en que no hay dato pedido y ninguno es un
    error: que el router no haya corrido la segunda pasada, que haya
    contestado `ninguno`, y que haya inventado un valor fuera del enum.
    Los tres terminan en el mismo lugar —la respuesta general de siempre—
    así que colapsarlos acá evita repetir el `if` en cada llamador.
    """
    texto = str(crudo or "").strip().lower()
    return DatoDeCorrida(texto) if texto in DatoDeCorrida._value2member_map_ else None


def contar_pasos_ionicos(oszicar: Optional[str]) -> Optional[int]:
    """Pasos iónicos de un OSZICAR. `None` si no hay archivo."""
    if not oszicar:
        return None
    return len(_PASO_IONICO_RE.findall(oszicar))


def nsw_de(incar: Optional[str]) -> Optional[int]:
    """El NSW declarado en un INCAR. `None` si no está."""
    if not incar:
        return None
    match = _NSW_RE.search(incar)
    return int(match.group(1)) if match else None


def ultima_energia(oszicar: Optional[str]) -> Optional[float]:
    """El último E0 de un OSZICAR. `None` si no hay ninguno legible."""
    if not oszicar:
        return None
    matches = _E0_RE.findall(oszicar)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def contar_atomos(poscar: Optional[str]) -> Optional[int]:
    """Total de átomos de un POSCAR/CONTCAR: la línea de cantidades es la
    7ª (VASP 5, con símbolos en la 6ª) o la 6ª (VASP 4)."""
    if not poscar:
        return None
    lines = poscar.splitlines()
    for idx in (6, 5):
        if len(lines) > idx:
            try:
                return sum(int(x) for x in lines[idx].split())
            except ValueError:
                continue
    return None
