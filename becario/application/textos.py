"""Textos que el bot le muestra a la persona.

Viven separados de `services.py` para que la fachada se lea como lógica y
no como copy. `services.py` los re-exporta: tests y presentación los
siguen importando desde ahí.
"""
from __future__ import annotations

from ..domain.models import RouterFailureReason, RouterUnavailableError

HELP_TEXT = (
    "❓ No pude interpretar tu pedido. Probá con:\n"
    '• "Relajá los parámetros de red del bulk de W"\n'
    '• "Hacé la curva de convergencia de ENCUT para Zr hcp"\n'
    '• "Dame los parámetros de red del cálculo del Zr"\n'
    '• "Mostrá el estado de mis trabajos"\n'
    '• "Cancelá el trabajo 12345"\n'
    '• "Consultá el historial"\n'
    '• "Generá un POSCAR de Si diamond 2x2x2"\n'
    '• "Creame la carpeta /home/usuario/pruebas"\n'
    '• "Qué archivos hay en /data/becario_runs"'
)

NOT_REGISTERED_TEXT = (
    "🚫 No estás registrado en B.E.C.A.R.I.O. Pedile a quien administra "
    "el grupo que te agregue con tu usuario del cluster."
)

# El modelo no llegó a pronunciarse. Son mensajes SEPARADOS de `HELP_TEXT`
# porque dicen algo distinto: ahí el pedido no se entendió y conviene
# reescribirlo; acá el pedido puede haber estado perfecto y no llegó ni a
# leerse. Y son tres y no uno porque llevan a acciones distintas: ante un
# timeout tiene sentido reintentar, ante un servidor caído hay que avisarle
# a alguien.
ROUTER_TIMEOUT_TEXT = (
    "⏱️ Tiempo de espera agotado: tardé demasiado interpretando tu pedido y "
    "corté la espera{limite}.\n\nNo es que no te haya entendido — no llegué "
    "a terminar de pensarlo. Probá de nuevo; si el pedido era largo, "
    "mandámelo más corto o partido en dos."
)
ROUTER_UNREACHABLE_TEXT = (
    "🔌 No puedo hablar con el modelo de lenguaje (no responde), así que "
    "ahora mismo no puedo interpretar pedidos.\n\nNo es tu mensaje: avisale "
    "a quien administra el bot. Tus trabajos en el cluster siguen corriendo "
    "igual."
)
ROUTER_ERROR_TEXT = (
    "⚠️ El modelo de lenguaje falló al interpretar tu pedido.\n\nNo es tu "
    "mensaje. Probá de nuevo en un momento; si sigue pasando, avisale a "
    "quien administra el bot."
)



def _router_failure_text(exc: RouterUnavailableError) -> str:
    """Traduce el fallo del router al mensaje que corresponde."""
    if exc.reason is RouterFailureReason.TIMEOUT:
        limite = (
            f" (más de {exc.timeout_seconds:g} segundos)"
            if exc.timeout_seconds
            else ""
        )
        return ROUTER_TIMEOUT_TEXT.format(limite=limite)
    if exc.reason is RouterFailureReason.UNREACHABLE:
        return ROUTER_UNREACHABLE_TEXT
    return ROUTER_ERROR_TEXT


FOREIGN_CONFIRMATION_TEXT = "⚠️ Esta confirmación no te pertenece."
# Se conserva el texto histórico para el caso genérico; los casos que sí se
# pueden distinguir tienen el suyo (ver
# `BecarioService._stale_confirmation_reply` en `services.py`).
EXPIRED_CONFIRMATION_TEXT = "⌛ Esta confirmación expiró o ya fue usada."
USED_CONFIRMATION_TEXT = (
    "✔️ Ese botón ya se usó — la acción se hizo con el primer toque. "
    "Si no viste la respuesta, pudo perderse en el camino: mirá el chat un "
    "momento antes de repetir el pedido."
)
ALREADY_MODIFYING_TEXT = (
    "✏️ Ya estás modificando ese plan: apretaste ✏️ y quedó esperando tu "
    "cambio. Decímelo y sigo (p. ej. «usá 2 nodos», «subí el ENCUT a 600»), "
    "o escribí «cancelar» para descartarlo."
)
