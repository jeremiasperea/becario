"""Adaptador de enrutamiento con Ollama structured outputs.

Gemma no soporta tool calling nativo en Ollama, pero los *structured
outputs* (parámetro `format` con un JSON Schema) funcionan con cualquier
modelo y obligan al LLM a emitir JSON conforme al schema. El schema se
deriva automáticamente del modelo Pydantic `RouterDecision` — una sola
fuente de verdad para el contrato con el LLM.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..domain.models import Intent, RoutedRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrato con el LLM (define el JSON Schema de la respuesta)
# ---------------------------------------------------------------------------


class RouterParams(BaseModel):
    """Parámetros que el LLM puede extraer del mensaje. Todos opcionales;
    la validación fuerte ocurre después, en los modelos de dominio."""

    job_id: Optional[str] = None
    nombre_trabajo: Optional[str] = None
    particion: Optional[str] = None
    nodos: Optional[int] = None
    tiempo_limite: Optional[str] = None
    script_remoto: Optional[str] = None
    filtro_busqueda: Optional[str] = None
    # estructura atómica:
    formula: Optional[str] = None
    tipo_estructura: Optional[str] = Field(
        default=None, description="bulk o molecule"
    )
    red_cristalina: Optional[str] = Field(
        default=None, description="diamond, fcc, bcc, hcp, rocksalt, zincblende…"
    )
    parametro_red: Optional[float] = Field(default=None, description="en Å")
    supercelda: Optional[list[int]] = Field(
        default=None, description="[nx, ny, nz]"
    )
    vacio: Optional[float] = Field(default=None, description="en Å")
    formato_salida: Optional[str] = Field(
        default=None, description="vasp, cif o xyz"
    )
    destino_remoto: Optional[str] = Field(
        default=None, description="directorio absoluto en el cluster"
    )


class RouterDecision(BaseModel):
    """Lo único que el LLM puede responder."""

    action: Intent
    parametros: RouterParams = Field(default_factory=RouterParams)


_SYSTEM_PROMPT = (
    "Sos el enrutador de B.E.C.A.R.I.O., un asistente HPC para simulación "
    "computacional de materiales. Analizá el mensaje del usuario y decidí "
    "la acción:\n"
    "- 'modificar_estructura': crear/generar estructuras atómicas "
    "(bulk, moléculas, superceldas, POSCAR para VASP)\n"
    "- 'enviar_slurm': lanzar/correr un cálculo en el cluster\n"
    "- 'consultar_db': buscar en el historial de cálculos\n"
    "- 'revisar_estado': estado de trabajos en cola (squeue/sacct)\n"
    "- 'cancelar_calculo': cancelar un trabajo\n"
    "- 'error': si el pedido no encaja en ninguna\n"
    "Extraé en 'parametros' solo los datos presentes en el mensaje. "
    "No inventes valores."
)


class OllamaRouter:
    """Router basado en structured outputs de Ollama (Gemma-compatible)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4:12b",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._schema = RouterDecision.model_json_schema()

    def route(self, user_text: str) -> RoutedRequest:
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    "stream": False,
                    "format": self._schema,  # <- structured output
                    "options": {"temperature": 0.0},
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw = response.json().get("message", {}).get("content", "")
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("Ollama no disponible: %s", exc)
            return RoutedRequest(intent=Intent.UNKNOWN, params={"motivo": str(exc)})

        return self.parse_llm_output(raw)

    @staticmethod
    def parse_llm_output(raw: str) -> RoutedRequest:
        """Validación final con Pydantic. Con structured outputs esto casi
        nunca falla, pero seguimos siendo defensivos (público para testear)."""
        try:
            decision = RouterDecision.model_validate_json(raw or "")
        except ValidationError:
            logger.warning("LLM devolvió JSON fuera de schema: %r", (raw or "")[:200])
            return RoutedRequest(intent=Intent.UNKNOWN, params={"raw": raw})
        params = decision.parametros.model_dump(exclude_none=True)
        return RoutedRequest(intent=decision.action, params=params)
