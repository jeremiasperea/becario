"""Estado en proceso de las ediciones de plan pendientes.

Separado de `services.py`: es el default del `PendingEditStore` y el
reconocimiento de «cancelar» mientras un plan espera su cambio.
`services.py` re-exporta ambos nombres.
"""
from __future__ import annotations

import re
import threading
from typing import Optional

from ..domain.models import PendingEdit


class _InMemoryPendingEdits:
    """Default en proceso del `PendingEditStore`.

    Vive acá y no en `infrastructure` a propósito: es un `dict` con un
    lock —sin I/O ni dependencias externas—, y que la aplicación importara
    de infraestructura para tener un default daría vuelta la dirección de
    las dependencias por comodidad.

    Sirve para los tests y para cualquier armado sin composition root.
    Producción inyecta `SQLitePendingEditStore`: acá un reinicio se lleva
    la repregunta en silencio, que es justo lo que se vino a arreglar.
    """

    def __init__(self) -> None:
        self._items: dict[int, PendingEdit] = {}
        self._lock = threading.Lock()

    def put(self, user_id: int, edit: PendingEdit) -> None:
        with self._lock:
            self._items[user_id] = edit

    def get(self, user_id: int) -> Optional[PendingEdit]:
        with self._lock:
            return self._items.get(user_id)

    def pop(self, user_id: int) -> Optional[PendingEdit]:
        with self._lock:
            return self._items.pop(user_id, None)

    def has(self, user_id: int) -> bool:
        with self._lock:
            return user_id in self._items

    def pop_expired(self, ttl_seconds: float) -> list[tuple[int, PendingEdit]]:
        with self._lock:
            vencidos = [(u, e) for u, e in self._items.items()
                        if e.expired(ttl_seconds)]
            for user_id, _ in vencidos:
                del self._items[user_id]
        return vencidos


_CANCEL_RE = re.compile(r"\A\s*(cancel|descart|olvid|dejal|no,?\s*(dejalo|nada))", re.IGNORECASE)


def _is_cancel_message(text: str) -> bool:
    """¿El mensaje pide descartar el plan en edición? («cancelar»,
    «cancelalo», «descartá todo», «olvidalo»…)."""
    return bool(_CANCEL_RE.match(text or ""))
