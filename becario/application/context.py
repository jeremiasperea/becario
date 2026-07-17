"""Objetos compartidos entre la fachada de casos de uso y sus handlers.

`Reply` y `_Ctx` viven acá (y no en `services`) para que tanto la fachada
como los módulos de `handlers` puedan importarlos sin ciclos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..domain.models import ClusterIdentity
from ..domain.ports import ClusterGateway


@dataclass(frozen=True)
class Reply:
    """Respuesta neutral respecto del canal de salida."""

    text: str
    needs_confirmation: bool = False
    confirmation_token: Optional[str] = None
    # Pide fuente de ancho fijo (tablas): cada canal decide cómo lograrlo.
    monospace: bool = False
    # La acción pendiente admite "modificar el plan" (tercer botón).
    allow_modify: bool = False
    # ¿Este paso se pudo completar? Solo lo usa `PlanExecutor` (vía
    # `_materialize_step`) para decidir si un plan multi-paso corta acá;
    # el canal de salida lo ignora. Default True: un paso de un solo
    # intent (el camino de hoy) nunca lo consulta.
    ok: bool = True


@dataclass(frozen=True)
class _Ctx:
    """Contexto resuelto de un pedido: quién es y con qué cuenta opera."""

    chat_id: int
    user_id: int
    identity: ClusterIdentity
    cluster: ClusterGateway
    # Id en el RouterDecisionLog de la decisión que originó este pedido;
    # viaja hasta el PendingPlan para que confirmar/cancelar etiquete el
    # ruteo. None cuando el log está apagado o el pedido no vino del router.
    decision_id: Optional[int] = None
