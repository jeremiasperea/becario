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

from ..domain.models import Intent, Plan, PlanStep

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
    # cálculo VASP completo:
    tipo_calculo: Optional[str] = Field(
        default=None, description="relajacion, estatico o convergencia_encut"
    )
    encut: Optional[int] = Field(default=None, description="ENCUT en eV")
    encut_min: Optional[int] = Field(default=None, description="inicio del barrido de ENCUT, en eV")
    encut_max: Optional[int] = Field(default=None, description="fin del barrido de ENCUT, en eV")
    encut_paso: Optional[int] = Field(default=None, description="paso del barrido de ENCUT, en eV")
    puntos_k: Optional[list[int]] = Field(
        default=None, description="grilla de k-points [kx, ky, kz]"
    )


# Docstrings de RouterStep/RouterDecision se mantienen cortos a propósito:
# Pydantic los vuelca como "description" en el JSON Schema que se le manda
# al LLM, y ese schema tiene presupuesto de tamaño (ver ADR-0006 y el gate
# de tarea 3.1/3.2 en tests/test_router_parsing.py). La explicación larga
# vive acá, en comentarios normales, no en el docstring.
#
# RouterStep: un paso del plan tal como lo emite el LLM. Distinto del
# `PlanStep` de dominio: acá `parametros` es `RouterParams` (la superficie
# de extracción cruda), no el `dict` que ya validó/limpió el dominio.
# `parse_llm_output` convierte uno en el otro.
class RouterStep(BaseModel):
    """Un paso del plan (contrato del LLM)."""

    action: Intent
    parametros: RouterParams = Field(default_factory=RouterParams)


# RouterDecision: lo único que el LLM puede responder, un plan de 1 a 5
# pasos. `RouterParams` se referencia una sola vez vía `$ref` (JSON-Schema
# `$defs`) y la reutiliza cada paso — no se duplica el schema de
# parámetros por paso.
class RouterDecision(BaseModel):
    """Plan de 1 a 5 pasos."""

    steps: list[RouterStep] = Field(min_length=1, max_length=5)


_SYSTEM_PROMPT = (
    "Sos el enrutador de B.E.C.A.R.I.O., un asistente HPC para simulación "
    "computacional de materiales. Analizá el mensaje del usuario y decidí "
    "la acción:\n"
    "- 'modificar_estructura': solo crear/generar archivos de estructuras "
    "atómicas (bulk, moléculas, superceldas, POSCAR para VASP)\n"
    "- 'preparar_calculo': preparar y correr un cálculo DFT/VASP completo. "
    "tipo_calculo: 'relajacion' (relajar/optimizar/minimizar estructura o "
    "parámetros de red), 'estatico' (energía de un punto), o "
    "'convergencia_encut' (curva/barrido/convergencia de ENCUT o del cutoff)\n"
    "- 'enviar_slurm': lanzar un script de cálculo que YA existe en el cluster\n"
    "- 'consultar_db': SOLO historial de trabajos/cálculos pasados (fechas, "
    "nombres y estados) — nunca archivos ni carpetas\n"
    "- 'consultar_resultados': resultados FÍSICOS de un cálculo ya hecho: "
    "parámetros de red, celda relajada, energía\n"
    "- 'revisar_estado': estado de trabajos en cola (squeue/sacct)\n"
    "- 'cancelar_calculo': cancelar un trabajo\n"
    "- 'crear_directorio': crear una carpeta/directorio en el cluster; "
    "extraé la ruta en destino_remoto\n"
    "- 'listar_archivos': mostrar archivos, carpetas o la estructura de "
    "directorios del cluster; la ruta va en destino_remoto (puede faltar)\n"
    "- 'error': si el pedido no encaja en ninguna\n"
    "Ejemplos:\n"
    "'dame los parámetros de red del cálculo del zirconio bulk' -> "
    "consultar_resultados, formula=Zr\n"
    "'qué energía dio la relajación del W' -> consultar_resultados, "
    "formula=W\n"
    "'mostrá el historial de cálculos' -> consultar_db\n"
    "'minimizá los parámetros de red del bulk de W' -> preparar_calculo, "
    "tipo_calculo=relajacion, formula=W\n"
    "'curva de convergencia de ENCUT para Zr hcp de 250 a 450' -> "
    "preparar_calculo, tipo_calculo=convergencia_encut, formula=Zr, "
    "red_cristalina=hcp, encut_min=250, encut_max=450\n"
    "'corré el script /home/ana/run.sh' -> enviar_slurm, "
    "script_remoto=/home/ana/run.sh\n"
    "'generá un POSCAR de Si diamond 2x2x2' -> modificar_estructura\n"
    "'creame la carpeta /home/ana/pruebas' -> crear_directorio, "
    "destino_remoto=/home/ana/pruebas\n"
    "'mostrame la estructura de archivos del cluster' -> listar_archivos\n"
    "'qué archivos hay en /data/becario_runs' -> listar_archivos, "
    "destino_remoto=/data/becario_runs\n"
    "Si el mensaje pide más de una acción, emitilas en 'steps', en el "
    "mismo orden en que las pidió el usuario, cada paso con su propia "
    "'action' y 'parametros':\n"
    "'creá la carpeta /home/ana/x y después listá /home/ana/y' -> "
    "paso 1: crear_directorio, destino_remoto=/home/ana/x; "
    "paso 2: listar_archivos, destino_remoto=/home/ana/y\n"
    "'generá el POSCAR de Si y creá la carpeta /home/ana/run' -> "
    "paso 1: modificar_estructura, formula=Si; "
    "paso 2: crear_directorio, destino_remoto=/home/ana/run\n"
    "Extraé en 'parametros' solo los datos presentes en el mensaje. "
    "No inventes valores. En 'formula' usá siempre el símbolo químico "
    "(zirconio->Zr, tungsteno/wolframio->W, silicio->Si)."
)


_EDIT_PROMPT = (
    "Sos el extractor de cambios de B.E.C.A.R.I.O., un asistente HPC. El "
    "usuario ya tiene un cálculo armado y este mensaje describe UN CAMBIO "
    "sobre ese plan. Extraé únicamente los parámetros que el mensaje "
    "menciona (nodos, particion, tiempo_limite, encut, encut_min, "
    "encut_max, encut_paso, puntos_k, supercelda, formula, red_cristalina, "
    "parametro_red…). No inventes valores ni completes los que no menciona.\n"
    "Ejemplos:\n"
    "'cambiá a 2 nodos' -> nodos=2\n"
    "'subí el ENCUT máximo a 600' -> encut_max=600\n"
    "'usá la partición gpu y 4 horas' -> particion=gpu, tiempo_limite=04:00:00"
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
        self._params_schema = RouterParams.model_json_schema()

    def _chat(self, system_prompt: str, user_text: str, schema: dict) -> Optional[str]:
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    "stream": False,
                    "format": schema,  # <- structured output
                    "options": {"temperature": 0.0},
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except (httpx.HTTPError, ValueError) as exc:
            logger.error("Ollama no disponible: %s", exc)
            return None

    def route(self, user_text: str) -> Plan:
        raw = self._chat(_SYSTEM_PROMPT, user_text, self._schema)
        if raw is None:
            return Plan(steps=[PlanStep(action=Intent.UNKNOWN)])
        return self.parse_llm_output(raw)

    def extract_params(self, user_text: str) -> dict:
        """Solo parámetros (sin acción): para mensajes que modifican un
        plan ya armado."""
        raw = self._chat(_EDIT_PROMPT, user_text, self._params_schema)
        if raw is None:
            return {}
        try:
            params = RouterParams.model_validate_json(raw or "")
        except ValidationError:
            logger.warning("Extracción de cambio fuera de schema: %r", (raw or "")[:200])
            return {}
        return params.model_dump(exclude_none=True)

    @staticmethod
    def parse_llm_output(raw: str) -> Plan:
        """Validación final con Pydantic, fail-closed (R1-003 del diseño).

        Con structured outputs el JSON casi nunca sale mal formado, pero
        seguimos siendo defensivos (público para testear). Cualquier
        violación — JSON fuera de schema O forma de plan inválida
        (más de un paso destructivo, destructivo no al final, fuera del
        rango 1-5 pasos) — rechaza el plan ENTERO: nunca se trunca,
        reordena ni se ejecuta parcialmente."""
        try:
            decision = RouterDecision.model_validate_json(raw or "")
            steps = [
                PlanStep(
                    action=step.action,
                    parametros=step.parametros.model_dump(exclude_none=True),
                )
                for step in decision.steps
            ]
            return Plan(steps=steps)
        except ValidationError:
            logger.warning("LLM devolvió JSON fuera de schema: %r", (raw or "")[:200])
            return Plan(steps=[PlanStep(action=Intent.UNKNOWN, parametros={"raw": raw})])
