"""Reintento con backoff exponencial y jitter, en un solo lugar.

Uno por adaptador era la alternativa, y es la que termina con tres políticas
distintas que nadie comparó nunca. Acá está una, con los dos parámetros que
importan (cuántos intentos y qué cuenta como transitorio) en manos de quien
llama — porque eso SÍ cambia entre un `sacct` y una llamada al LLM.

Módulo hoja: no importa nada del proyecto.

Sobre qué se puede reintentar
-----------------------------
Este helper no sabe si la operación es segura de repetir, y no puede
saberlo. Esa decisión es de quien llama, y es la parte peligrosa: `sbatch`
NO es idempotente. Si el envío llega al cluster y la respuesta se pierde en
el camino, reintentar manda el trabajo dos veces — y el usuario paga dos
veces las horas de cómputo por un error que nunca vio.

Por eso el default en el gateway es no reintentar, y cada operación que sí
lo hace lo declara en el call site, a la vista.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def dormir_backoff(segundos: float) -> None:
    """La pausa entre reintentos, aislada en una función a propósito.

    Es una costura para los tests: parchear `time.sleep` a secas apaga
    también las esperas legítimas de otros tests (los de vencimiento de TTL,
    por ejemplo, que necesitan que el reloj avance de verdad). Parchear esto
    apaga exactamente el backoff y nada más.
    """
    time.sleep(segundos)


def espera_con_jitter(
    intento: int, base: float, tope: float, azar: Callable[[], float] = random.random
) -> float:
    """Backoff exponencial con *full jitter*: un valor al azar entre 0 y el
    tope del intento.

    El jitter no es adorno. Sin él, N clientes que fallan por la misma causa
    (el cluster que se reinició) vuelven todos juntos en el mismo instante,
    justo cuando menos aguanta. Acá hay un proceso por persona del grupo
    apuntando al mismo nodo de login, así que la manada es chica pero existe.
    """
    return azar() * min(tope, base * (2 ** max(0, intento - 1)))


def reintentar(
    operacion: Callable[[], T],
    *,
    intentos: int = 3,
    resultado_transitorio: Callable[[T], bool] = lambda _r: False,
    excepcion_transitoria: Callable[[BaseException], bool] = lambda _e: False,
    base: float = 0.5,
    tope: float = 8.0,
    etiqueta: str = "operación",
    dormir: Optional[Callable[[float], None]] = None,
    azar: Callable[[], float] = random.random,
) -> T:
    """Corre `operacion` hasta `intentos` veces mientras el fallo sea
    transitorio.

    Sirve para las dos formas de fallar que conviven en el proyecto: las que
    DEVUELVEN el fallo (`CommandResult(ok=False)`) y las que lo LEVANTAN
    (httpx, mp-api). De ahí los dos predicados.

    `dormir` y `azar` se inyectan para que los tests no esperen de verdad ni
    dependan del azar. Con `dormir=None` se resuelve `dormir_backoff` en el
    momento de la llamada, no al definir la función: si fuera un default
    ligado, parchear el módulo desde un test no tendría efecto porque el
    default ya tendría capturada la función original.
    """
    esperar = dormir if dormir is not None else dormir_backoff
    ultimo_error: BaseException | None = None
    for intento in range(1, intentos + 1):
        try:
            resultado = operacion()
        except BaseException as exc:  # noqa: BLE001 - se re-levanta abajo
            if not excepcion_transitoria(exc) or intento == intentos:
                raise
            ultimo_error = exc
        else:
            if not resultado_transitorio(resultado) or intento == intentos:
                if intento > 1:
                    logger.info("%s salió bien al intento %s", etiqueta, intento)
                return resultado

        pausa = espera_con_jitter(intento, base, tope, azar)
        logger.warning(
            "%s falló (intento %s/%s%s); reintento en %.2f s",
            etiqueta, intento, intentos,
            f": {ultimo_error}" if ultimo_error else "", pausa,
        )
        esperar(pausa)

    # Inalcanzable: el último intento sale por `return` o por `raise`.
    raise AssertionError("reintentar() terminó el bucle sin resolver")  # pragma: no cover
