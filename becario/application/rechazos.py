"""El rastro de los tres «no puedo» del bot.

El bot rechaza un pedido en tres momentos distintos del pipeline, y cada
uno recibe algo distinto: el texto crudo, la salida del router, o el plan
ya parseado. Por eso son tres y tienen que seguir siendo tres —— en el
momento en que corre cada uno, la información de los otros dos todavía no
existe:

* `PRE_ROUTER`   `fuera_de_alcance(texto)` mira el mensaje tal como llegó.
                 Corre ANTES de rutear, que es lo único que lo hace útil:
                 corta gratis un pedido imposible en vez de gastar la
                 descomposición para llegar a un plan que no sirve.
* `ROUTER`       El router se pronunció y no hay handler para lo que dijo
                 (`Intent.UNKNOWN`, o cualquier acción sin handler). Recién
                 se sabe DESPUÉS de rutear.
* `VOCABULARIO`  El plan es válido, pero el dato pedido no está en el
                 vocabulario y el modelo se abstuvo. Necesita los params
                 ya parseados.

Lo que sí tienen en común es que hay que poder RASTREARLOS. Cuando alguien
pega un mensaje del bot y pregunta «¿por qué me contestó esto?», la
respuesta depende de cuál de los tres portones se cerró — y hasta acá eso
solo se podía deducir reconociendo la prosa de memoria: uno dejaba su
propio `logger.info`, otro quedaba enterrado en el `steps_json` del
decision log, y el tercero no dejaba rastro ninguno.

Este módulo NO es un cuarto portón. No decide nada y no arma ningún texto:
cada portón sigue contestando exactamente lo que contestaba. Acá viven solo
los tres nombres de origen y la única línea que los registra, para que sean
una sola cosa grepeable y no puedan derivar por separado.

Para saber cuál fue, alcanza con buscar el evento:

    grep 'rechazo origen=' becario.log
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Los tres momentos en que se puede cerrar un pedido. Son valores fijos: el
# punto de tenerlos acá y no como literales sueltos en cada sitio es que un
# rename no pueda dejar dos portones hablando distinto.
PRE_ROUTER = "pre_router"
ROUTER = "router"
VOCABULARIO = "vocabulario"

ORIGENES = frozenset({PRE_ROUTER, ROUTER, VOCABULARIO})


def registrar_rechazo(
    origen: str,
    *,
    user_id: Optional[int] = None,
    detalle: str = "",
) -> None:
    """Deja constancia de que un pedido se rechazó, y en qué momento.

    `detalle` se recorta a 120 caracteres: sirve para reconocer el pedido en
    el log, no para archivarlo —— el mensaje entero del usuario ya viaja al
    `RouterDecisionLog` cuando el router llegó a pronunciarse.
    """
    logger.info("rechazo origen=%s user=%s: %.120s", origen, user_id, detalle)
