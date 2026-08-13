"""Identificador corto de un fallo inesperado, compartido por el chat y el log.

Existe por un motivo concreto: cuando algo se rompe de una forma que nadie
modeló, el usuario ve un mensaje y el operador ve un traceback, y hasta ahora
no había manera de saber que eran el mismo hecho. Un id de ocho caracteres
que aparece en los dos lados convierte «no me contestó» en una búsqueda.

Módulo hoja a propósito: no importa nada del proyecto, así que lo pueden usar
tanto la capa de aplicación como la de presentación sin dar vuelta la
dirección de las dependencias.
"""
from __future__ import annotations

import uuid

# Texto para un fallo que ocurrió ANTES de tocar el cluster: no se ejecutó
# nada, y repetir el pedido es seguro.
FALLO_INESPERADO = (
    "⚠️ Se me rompió algo procesando tu pedido y no ejecuté nada. "
    "Podés volver a intentarlo. Si sigue pasando, pasale este código a "
    "quien administre el bot: incidente {incidente}"
)

# Texto para un fallo DESPUÉS de que el usuario confirmó una acción. Es otro
# mensaje porque es otra situación: el token ya se consumió y la acción pudo
# haberse ejecutado, así que la instrucción correcta es "verificá", no
# "reintentá". Un `sbatch` repetido a ciegas son dos trabajos en la cola.
FALLO_TRAS_CONFIRMAR = (
    "⚠️ Se me rompió algo ejecutando la acción que confirmaste, y no puedo "
    "asegurarte si llegó a hacerse. **No la repitas a ciegas**: fijate "
    "primero con «estado de mis trabajos». Código del incidente: {incidente}"
)


def nuevo_incidente() -> str:
    """Id corto y único. Ocho caracteres hex alcanzan de sobra para
    correlacionar dentro de un log, y son pocos para copiarlos a mano
    desde el celular — que es donde va a estar el usuario cuando pase."""
    return uuid.uuid4().hex[:8]
