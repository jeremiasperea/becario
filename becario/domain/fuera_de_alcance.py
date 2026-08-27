"""Pedidos que el bot NO sabe hacer, reconocidos antes de intentarlos.

Existe por un pedido real: «Arma una supercelda de Zr sobre W 2x2/3x3
respectivamente, no olvides agregas 15 ang de vacío en z». El bot contestó
con un batch de ocho pasos, prolijo y numerado, listo para aprobar de un
botón. Adentro había un `red_cristalina=bcc_fcc` que no es una red, un
`formula=Zr_on_W` que no es una fórmula, W declarado `fcc` cuando es bcc,
las superceldas cambiadas y los 15 Å de vacío perdidos en el camino.

Nada de eso era el problema de fondo. **El bot no sabe apilar dos
materiales**: una heterostructura no está en `StructureKind` ni en ningún
lado. El pedido era imposible desde el principio, y en vez de decirlo se
inventó una interpretación y la presentó con la confianza de una respuesta
buena.

Por eso el chequeo mira el TEXTO y corre ANTES del router. Dos razones:

1. Lo que hay que reconocer es la capacidad pedida, no el plan emitido.
   Sobre este mismo mensaje el modelo emitió pasos con `formula=Zr` y
   `formula=W` por separado, cada uno perfectamente válido: mirando el
   plan no hay nada que objetar, y sin embargo el pedido no se cumplió.
2. Es determinista y gratis. El pedido original era largo, así que iba a
   la descomposición —decenas de segundos de CPU— para terminar en un
   plan que no servía.

Es la misma decisión que ya se tomó con el barrido de k-points en las
sugerencias: **no ofrecer lo que no se sabe hacer.** Un «no sé» es una
respuesta útil; ocho pasos inventados no.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .models import is_plausible_formula

LO_QUE_SE_ARMAR = (
    "Sé armar, de a un material por vez:\n"
    '• un bulk — "generá un POSCAR de Si diamond 2x2x2"\n'
    '• una losa, diciéndome la cara — "armá el slab de Zr (001)"\n'
    '• una molécula — "una molécula de H2O con 12 Å de vacío"'
)

NO_SE_APILAR = (
    "🚫 No sé armar dos materiales juntos: una heterostructura, algo sobre "
    "un sustrato, una bicapa. No está entre las estructuras que puedo "
    "construir, así que prefiero decírtelo ahora y no armarte un plan que "
    "no es lo que pediste.\n\n" + LO_QUE_SE_ARMAR
)

# Una fórmula, tal como se escribe en un mensaje. Deliberadamente laxo: el
# filtro real es `is_plausible_formula`, que sabe de la tabla periódica.
_FORMULA = r"[A-Z][A-Za-z0-9]{0,15}"

# "Zr sobre W", "W on Si", "Zr encima del W". El artículo es opcional y el
# lado derecho tiene que arrancar en mayúscula, que es lo que separa
# "Zr sobre W" de "información sobre el cluster".
_APILADO_RE = re.compile(
    rf"\b({_FORMULA})\s+(?:sobre|encima\s+de|arriba\s+de|on)\s+"
    rf"(?:(?:el|la|los|las|un|una|del|de\s+la)\s+)?({_FORMULA})\b"
)

# Vocabulario que nombra la capacidad directamente. Se compara sin acentos
# pero CON los límites de palabra puestos: pegar el texto para comparar
# —como hace `_plain` con los nombres de red— convertiría "carpeta pilas"
# en un pedido de apilado.
#
# `interfaz` queda AFUERA a propósito: en castellano también es la interfaz
# de usuario, y el mismo chat trajo un «acá debería haber una opción de
# modificar» que no habla de física.
_VOCABULARIO_RE = re.compile(
    r"\b(?:hetero[\s-]?e?structur\w*|bicapa\w*|multicapa\w*"
    r"|sustrato\w*|substrato\w*|apil\w*|adsor\w*)\b"
)


def _sin_acentos(texto: str) -> str:
    """Minúsculas y sin tildes, conservando los espacios: el pedido llega
    dictado por voz tanto como escrito ("apilá" y "apila" son lo mismo)."""
    plano = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in plano if not unicodedata.combining(c))


def fuera_de_alcance(texto: str) -> Optional[str]:
    """El mensaje que explica por qué este pedido no se puede, o `None`.

    Conservador donde importa. Un falso positivo bloquea un pedido legítimo
    y es lo peor que puede pasar acá, así que el lado izquierdo y el derecho
    de "sobre" tienen que ser los dos fórmulas de verdad —no cualquier par
    de palabras— y el vocabulario se limita a términos que solo nombran esta
    capacidad. Cuando rebota, el mensaje dice qué SÍ se puede: sin eso, un
    falso positivo deja al usuario sin salida.
    """
    if not texto:
        return None
    if _VOCABULARIO_RE.search(_sin_acentos(texto)):
        return NO_SE_APILAR
    for encima, abajo in _APILADO_RE.findall(texto):
        if is_plausible_formula(encima) and is_plausible_formula(abajo):
            return NO_SE_APILAR
    return None
