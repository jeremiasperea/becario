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
# Validación de arranque: server reachability + presencia del modelo
# ---------------------------------------------------------------------------
#
# Estas excepciones señalan las dos formas en que el chequeo de arranque
# puede fallar. El adaptador NUNCA imprime ni hace `SystemExit`: eso es
# responsabilidad de `main.py` (composition root), que las captura y decide
# el mensaje + el código de salida (ver diseño, ADR-1/ADR-2).


class OllamaValidationError(RuntimeError):
    """Base de los errores de validación de arranque de Ollama."""


class OllamaUnreachableError(OllamaValidationError):
    """El servidor de Ollama no respondió o respondió con datos inválidos."""


class OllamaModelMissingError(OllamaValidationError):
    """El modelo configurado no está entre los modelos disponibles."""

    def __init__(self, model: str, available: list[str]) -> None:
        self.model = model
        self.available = available
        super().__init__(
            f"modelo {model!r} no encontrado; disponibles: {available!r}"
        )


# ---------------------------------------------------------------------------
# Contrato con el LLM (define el JSON Schema de la respuesta)
# ---------------------------------------------------------------------------


class RouterParams(BaseModel):
    """Parámetros que el LLM puede extraer del mensaje. Todos opcionales;
    la validación fuerte ocurre después, en los modelos de dominio."""

    job_id: Optional[str] = Field(
        default=None, description="id numérico del trabajo en SLURM"
    )
    nombre_trabajo: Optional[str] = Field(
        default=None, description="nombre del trabajo/cálculo"
    )
    particion: Optional[str] = Field(
        default=None, description="partición (cola) de SLURM"
    )
    nodos: Optional[int] = Field(default=None, description="cantidad de nodos")
    tiempo_limite: Optional[str] = Field(
        default=None, description="límite de tiempo, HH:MM:SS o D-HH:MM:SS"
    )
    script_remoto: Optional[str] = Field(
        default=None, description="ruta del script que YA existe en el cluster"
    )
    filtro_busqueda: Optional[str] = Field(
        default=None, description="texto por el que filtrar el historial"
    )
    # estructura atómica:
    formula: Optional[str] = Field(
        default=None, description="fórmula con símbolos químicos: Zr, W, ZrO2"
    )
    tipo_estructura: Optional[str] = Field(
        default=None, description="bulk, molecule o slab"
    )
    # Losa/superficie. `miller` NO se adivina: si el pedido habla de una
    # superficie y no lo trae, el servicio lo pide (guía en `_SYSTEM_PROMPT`).
    miller: Optional[list[int]] = Field(
        default=None, description="cara de la losa [h, k, l]"
    )
    capas: Optional[int] = Field(default=None, description="capas de la losa")
    eje_vacio: Optional[str] = Field(
        default=None, description="eje del vacío de la losa: x, y o z"
    )
    red_cristalina: Optional[str] = Field(
        default=None, description="diamond, fcc, bcc, hcp, rocksalt, zincblende…"
    )
    parametro_red: Optional[float] = Field(
        default=None, description="parámetro de red, en Å"
    )
    supercelda: Optional[list[int]] = Field(
        default=None, description="repeticiones de la celda [nx, ny, nz]"
    )
    vacio: Optional[float] = Field(
        default=None, description="espesor del vacío, en Å"
    )
    formato_salida: Optional[str] = Field(
        default=None, description="formato del archivo: vasp, cif o xyz"
    )
    destino_remoto: Optional[str] = Field(
        default=None,
        description="ruta absoluta en el cluster (directorio o archivo)",
    )
    nombre_archivo: Optional[str] = Field(
        default=None, description="nombre de UN archivo, sin ruta: CONTCAR, OSZICAR"
    )
    # Fuente de estructura (Materials Project). `mp_id` es el id explícito de
    # un polimorfo; `fuente_estructura` fuerza el origen. Cuándo usar cada uno
    # (y cuándo dejarlos vacíos) sigue en `_SYSTEM_PROMPT`: es una regla de
    # ruteo entre campos, no la definición de este campo.
    mp_id: Optional[str] = Field(
        default=None, description="id de Materials Project: mp-149"
    )
    fuente_estructura: Optional[str] = Field(
        default=None, description="ase, mp o relajado"
    )
    # cálculo VASP completo:
    tipo_calculo: Optional[str] = Field(
        default=None, description="relajacion, estatico, convergencia_encut o dos"
    )
    encut: Optional[int] = Field(default=None, description="ENCUT en eV")
    encut_min: Optional[int] = Field(default=None, description="inicio del barrido de ENCUT, en eV")
    encut_max: Optional[int] = Field(default=None, description="fin del barrido de ENCUT, en eV")
    encut_paso: Optional[int] = Field(default=None, description="paso del barrido de ENCUT, en eV")
    puntos_k: Optional[list[int]] = Field(
        default=None, description="grilla de k-points [kx, ky, kz]"
    )
    # UN campo para los 169 tags del INCAR, en vez de uno por tag: un campo
    # por feature no escala como diseño (ADR-0006), más allá de que el techo
    # del schema ya no sea la restricción que se creía.
    # El dominio valida contra el vocabulario del manual antes de emitir nada.
    tags_incar: Optional[dict[str, str]] = Field(
        default=None, description="tags del INCAR y su valor: {'ISMEAR': '0'}"
    )
    magnetico: Optional[bool] = Field(
        default=None, description="true si el usuario pide un cálculo magnético"
    )


# Docstrings de RouterStep/RouterDecision se mantienen cortos, pero YA NO
# por presupuesto: Pydantic los vuelca como "description" en el JSON Schema
# que se le manda al LLM, y ahí solo sirve lo que el LLM necesita para
# emitir bien. La explicación de diseño —para quien lee el código— vive
# acá, en comentarios normales, que no viajan a ningún lado.
#
# El techo del schema es un gate contra crecimiento desbocado, no un
# racionamiento de bytes: está medido en 8000 B (ADR-0006, actualización 2)
# y las `description` de campo se pagan sin problema.
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
    "'convergencia_encut' (curva/barrido/convergencia de ENCUT o del cutoff), "
    "o 'dos' (densidad de estados, DOS, PDOS, estados proyectados)\n"
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
    "- 'ver_archivo': mostrar el CONTENIDO de UN archivo. Si el usuario "
    "nombra el archivo (CONTCAR, OSZICAR, INCAR…) poné ese nombre en "
    "nombre_archivo; si da una ruta absoluta al archivo, poné esa ruta en "
    "destino_remoto\n"
    "- 'error': si el pedido no encaja en ninguna\n"
    "Ejemplos:\n"
    "'dame los parámetros de red del cálculo del zirconio bulk' -> "
    "consultar_resultados, formula=Zr\n"
    "'qué energía dio la relajación del W' -> consultar_resultados, "
    "formula=W\n"
    "'mostrá el historial de cálculos' -> consultar_db\n"
    "'minimizá los parámetros de red del bulk de W' -> preparar_calculo, "
    "tipo_calculo=relajacion, formula=W\n"
    "Losas/superficies: 'slab', 'superficie', 'lámina', 'capas' -> "
    "tipo_estructura=slab. La cara va en 'miller' ((001), '110', "
    "'plano 111' -> [0,0,1] / [1,1,0] / [1,1,1]); el espesor en 'capas'; "
    "una repetición 'NxM' del plano en 'supercelda' como [N,M,1]. Si el "
    "pedido es de una superficie y NO nombra la cara, dejá 'miller' sin "
    "completar: NUNCA la inventes.\n"
    "'armá un slab de ZrO2 (001) de 5 capas, 2x1' -> modificar_estructura, "
    "tipo_estructura=slab, formula=ZrO2, miller=[0,0,1], capas=5, "
    "supercelda=[2,1,1]\n"
    "'una superficie de Zr acostada, vacío en y' -> "
    "modificar_estructura, tipo_estructura=slab, formula=Zr, eje_vacio=y\n"
    "'curva de convergencia de ENCUT para Zr hcp de 250 a 450' -> "
    "preparar_calculo, tipo_calculo=convergencia_encut, formula=Zr, "
    "red_cristalina=hcp, encut_min=250, encut_max=450\n"
    "Si el usuario nombra tags del INCAR con su valor ('con ISMEAR=0', "
    "'poné LORBIT 11', 'usá PREC=Normal'), extraelos en 'tags_incar' como "
    "{'ISMEAR': '0'}. Solo los que el mensaje nombre EXPLÍCITAMENTE: no "
    "traduzcas un pedido en prosa a tags, y no inventes valores.\n"
    "'estático de Zr con ISMEAR=0 y SIGMA=0.05' -> preparar_calculo, "
    "tipo_calculo=estatico, formula=Zr, tags_incar={'ISMEAR':'0','SIGMA':'0.05'}\n"
    "En 'preparar_calculo' de un compuesto, si el usuario nombra una "
    "estructura de Materials Project por su id ('mp-149', 'usá mp-19770') "
    "extraé ese id en 'mp_id'. Si pide forzar el origen ('bajala de "
    "Materials Project' -> mp; 'armala con ASE', 'estructura ideal' -> "
    "ase) poné 'ase' o 'mp' en 'fuente_estructura'. Sin pistas, dejá "
    "ambos sin completar (el sistema decide la fuente).\n"
    "Si el pedido parte de un resultado propio anterior —'el bulk de ZrO2 "
    "RELAJADO', 'la estructura ya relajada', 'partí del CONTCAR', 'usá el "
    "resultado de la relajación'— poné 'relajado' en 'fuente_estructura'. "
    "Ojo: 'relajá X' es tipo_calculo=relajacion (lo que se VA a hacer); "
    "'de X relajado' es fuente_estructura=relajado (de dónde se PARTE), y "
    "pueden aparecer juntos.\n"
    "'armá un slab del ZrO2 relajado' -> tipo_estructura=slab, "
    "formula=ZrO2, fuente_estructura=relajado\n"
    "'relajá Fe2O3 con mp-19770' -> preparar_calculo, formula=Fe2O3, "
    "mp_id=mp-19770\n"
    "Si el usuario pide un cálculo MAGNÉTICO (con espín, polarizado en "
    "espín, spin-polarized, ferromagnético/antiferromagnético) poné "
    "magnetico=true. Si no lo menciona, dejá magnetico sin completar.\n"
    "'relajá el bulk de Fe con espín' -> preparar_calculo, formula=Fe, "
    "magnetico=true\n"
    "'corré el script /home/ana/run.sh' -> enviar_slurm, "
    "script_remoto=/home/ana/run.sh\n"
    "'generá un POSCAR de Si diamond 2x2x2' -> modificar_estructura\n"
    "'creame la carpeta /home/ana/pruebas' -> crear_directorio, "
    "destino_remoto=/home/ana/pruebas\n"
    "'dentro de Zr creá las carpetas bcc y fcc' -> "
    "paso 1: crear_directorio, destino_remoto=Zr/bcc; "
    "paso 2: crear_directorio, destino_remoto=Zr/fcc "
    "(rutas RELATIVAS: el sistema las ancla en la carpeta de corridas)\n"
    "'mostrame la estructura de archivos del cluster' -> listar_archivos\n"
    "'listá mi home' -> listar_archivos (sin destino_remoto: el sistema "
    "resuelve el home del usuario)\n"
    "'mostramelo en forma de tree' / 'mostrame todo' -> listar_archivos "
    "(sin destino_remoto: si el pedido refiere a lo anterior o solo pide "
    "un formato, el sistema muestra el árbol del workspace)\n"
    "'qué archivos hay en /data/becario_runs' -> listar_archivos, "
    "destino_remoto=/data/becario_runs\n"
    "'mostrame el CONTCAR' -> ver_archivo, nombre_archivo=CONTCAR\n"
    "'ver el contenido de /home/ana/run/OSZICAR' -> ver_archivo, "
    "destino_remoto=/home/ana/run/OSZICAR\n"
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
    "No inventes valores. Las rutas de los ejemplos de arriba "
    "(/home/ana/..., /data/...) son ILUSTRATIVAS: jamás las copies a tu "
    "respuesta. Una ruta absoluta (con /) solo es válida si el usuario "
    "la escribió así en SU mensaje. Si nombra carpetas sin ruta absoluta "
    "('dentro de Zr', 'la carpeta pruebas'), emití la ruta RELATIVA tal "
    "cual (Zr/bcc) — nunca le agregues '/', '/home/…' ni ningún prefijo "
    "inventado. Si habla de 'mi home', 'mis corridas' o no da ruta, dejá "
    "destino_remoto sin completar; JAMÁS uses '/' (la raíz) como destino. "
    "En 'formula' usá siempre el símbolo químico (zirconio->Zr, "
    "tungsteno/wolframio->W, silicio->Si)."
)


class _Decomposition(BaseModel):
    """Contrato del descompositor: instrucciones simples, en orden."""

    instrucciones: list[str] = Field(default_factory=list)


_DECOMPOSE_PROMPT = (
    "Sos el descompositor de pedidos de B.E.C.A.R.I.O., un asistente HPC. "
    "Dividí el pedido del usuario en instrucciones SIMPLES y "
    "AUTO-CONTENIDAS, una por acción, en el orden pedido. Cada "
    "instrucción debe repetir TODOS los datos que le corresponden "
    "(material/fórmula, red cristalina, tipo de cálculo, carpeta), sin "
    "pronombres ni referencias a otras instrucciones: cada una tiene que "
    "entenderse sola. No inventes datos que el usuario no dijo. Máximo 5 "
    "instrucciones.\n"
    "Ejemplo: 'creá la carpeta W y relajá el bulk de W para bcc y fcc' ->\n"
    "1. creá la carpeta W\n"
    "2. relajá el bulk de W con red cristalina bcc, en la carpeta W/bcc\n"
    "3. relajá el bulk de W con red cristalina fcc, en la carpeta W/fcc"
)


# Segunda pasada de extracción de estructura. Cuando el mensaje trae otro
# eje de extracción, todos los modelos locales sueltan formula: bajo
# preparar_calculo (medido: qwen2.5:7b, gemma4:e4b y gemma3:4b, 0/3) y en
# los pedidos de losa, donde la cara/capas/supercelda se llevan la atención
# (medido: qwen2.5:7b, gemma4:12b y qwen2.5-coder:14b, 0/3 — los dos
# grandes sí sacan miller/capas/supercelda, pero ninguno el material).
# Sin el framing de intent lo extraen. Este prompt no clasifica ni menciona
# acciones: pide SOLO el material.
_STRUCT_PROMPT = (
    "Sos el extractor de estructura de B.E.C.A.R.I.O., un asistente HPC. "
    "El mensaje pide un cálculo DFT/VASP sobre una estructura atómica, o "
    "generar el archivo de una estructura atómica. "
    "Extraé SOLO el material: 'formula' (símbolo "
    "químico: zirconio->Zr, tungsteno/wolframio->W, silicio->Si) y "
    "'red_cristalina' (bcc, fcc, hcp, diamond…). No inventes valores; si el "
    "mensaje no menciona uno, dejalo sin completar.\n"
    "Ejemplos:\n"
    "'relajá el bulk de W bcc' -> formula=W, red_cristalina=bcc\n"
    "'convergencia de ENCUT para Zr hcp' -> formula=Zr, red_cristalina=hcp\n"
    "'armá un slab de ZrO2 (001) de 5 capas' -> formula=ZrO2"
)


# Intents cuyo handler necesita el material para poder hacer algo. Los dos
# lo repreguntan si falta, así que un backfill que no encuentra nada no
# rompe: solo deja el pedido como estaba.
_STRUCTURE_INTENTS = frozenset({Intent.PREPARE_CALC, Intent.MODIFY_STRUCTURE})

# Lo único que la segunda pasada tiene permitido devolver. El prompt pide
# estas dos y nada más, pero el modelo igual inventa parámetros de cálculo
# cuando el mensaje habla de una losa (medido: encut, puntos_k, tags_incar
# aparecidos de la nada). Filtrar acá evita que esa basura entre al plan.
_STRUCTURE_KEYS = ("formula", "red_cristalina")


_EDIT_PROMPT = (
    "Sos el extractor de cambios de B.E.C.A.R.I.O., un asistente HPC. El "
    "usuario ya tiene un cálculo armado y este mensaje describe UN CAMBIO "
    "sobre ese plan. Extraé únicamente los parámetros que el mensaje "
    "menciona (nodos, particion, tiempo_limite, encut, encut_min, "
    "encut_max, encut_paso, puntos_k, supercelda, formula, red_cristalina, "
    "parametro_red, miller, capas, eje_vacio…). No inventes valores ni "
    "completes los que no menciona.\n"
    "Ejemplos:\n"
    "'cambiá a 2 nodos' -> nodos=2\n"
    "'subí el ENCUT máximo a 600' -> encut_max=600\n"
    "'usá la partición gpu y 4 horas' -> particion=gpu, tiempo_limite=04:00:00\n"
    "'la (111)' -> miller=[1,1,1]\n"
    "'que sean 8 capas' -> capas=8\n"
    # La enumeración de campos de arriba NO alcanza: medido, el modelo
    # devolvía {} para 'fluorita a=5.07' mientras acertaba miller y capas,
    # que son los que tenían ejemplo. Es la respuesta a la repregunta de la
    # red, así que sin esto la repregunta no sirve para nada.
    "'fluorita a=5.07' -> red_cristalina=fluorita, parametro_red=5.07\n"
    "'rocksalt con parámetro 4.21' -> red_cristalina=rocksalt, "
    "parametro_red=4.21\n"
    # Respuesta a la repregunta de fase de un compuesto. Va con ejemplo por
    # la misma razón que la red: enumerar el campo no alcanza.
    "'la tetragonal' -> red_cristalina=tetragonal\n"
    "'monoclínica' -> red_cristalina=monoclinica"
)


# EditDecision/extract_edit (tarea 5.1): a diferencia de `extract_params`
# (un solo paso, sin ambigüedad posible), un plan de VARIOS pasos necesita
# saber a CUÁL apunta el cambio. `target_index` es 1-based y coincide con
# la enumeración que ya ve el usuario en la confirmación. El schema se
# mantiene liviano: un solo entero nuevo + `RouterParams` reusado por
# `$ref` (no se duplica) — mismo criterio que `RouterDecision` (ver
# diseño §2.2/§3.1, ADR-0006).
class EditDecision(BaseModel):
    """Salida de `extract_edit`: a qué paso (1-based) apunta el cambio,
    si el LLM tiene confianza, más el delta de parámetros."""

    target_index: Optional[int] = None
    parametros: RouterParams = Field(default_factory=RouterParams)


_EDIT_TARGET_PROMPT_TEMPLATE = (
    "Sos el extractor de cambios de B.E.C.A.R.I.O., un asistente HPC. El "
    "usuario tiene un plan pendiente de VARIOS pasos y este mensaje "
    "describe UN CAMBIO sobre UNO de ellos. El plan actual es:\n"
    "{plan_context}\n"
    "Decidí a qué paso (1-based, 'target_index') se aplica el cambio: si "
    "el mensaje dice explícitamente 'paso N' usá ese N; si no, deducilo "
    "del contenido del pedido (p. ej. mencionar una partición o nodos "
    "apunta al paso que envía o calcula, no a uno que solo crea una "
    "carpeta). Si NO estás seguro de a qué paso se refiere, dejá "
    "'target_index' en null — NUNCA adivines. Extraé en 'parametros' solo "
    "los datos que el mensaje menciona; no inventes valores.\n"
    "Ejemplos:\n"
    "'paso 2: usá 4 nodos' -> target_index=2, nodos=4\n"
    "'subí el ENCUT máximo a 600' (si un solo paso calcula algo) -> "
    "target_index del paso que calcula, encut_max=600\n"
    "'cambiá algo' (sin pistas de a qué paso) -> target_index=null"
)


def _strip_schema_titles(schema):
    """Poda recursiva de los `title` del JSON Schema.

    Pydantic genera un `title` por campo y por modelo ("Job Id",
    "RouterParams"…) que no aporta nada al LLM — el nombre ya está en la
    key — pero pesa contra el presupuesto de tamaño del schema
    (ADR-0006): ~590 bytes solo de titles en RouterDecision. Se aplica a
    todo schema que viaja a Ollama; las `description` quedan intactas."""
    if isinstance(schema, dict):
        return {
            k: _strip_schema_titles(v)
            for k, v in schema.items()
            if k != "title"
        }
    if isinstance(schema, list):
        return [_strip_schema_titles(item) for item in schema]
    return schema


def compact_json_schema(model: type[BaseModel]) -> dict:
    """El JSON Schema de `model` como se le manda a Ollama: sin titles."""
    return _strip_schema_titles(model.model_json_schema())


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
        self._schema = compact_json_schema(RouterDecision)
        self._params_schema = compact_json_schema(RouterParams)
        self._edit_decision_schema = compact_json_schema(EditDecision)

    @staticmethod
    def _model_matches(configured: str, available: list[str]) -> bool:
        """Regla de matching de modelo contra `GET /api/tags` (ADR-3).

        `/api/tags` siempre devuelve entradas `name:tag` completas. Si
        `configured` ya trae un tag explícito, se exige coincidencia exacta
        (evita que `gemma4:2b` conforme a un `gemma4:12b` configurado). Si
        `configured` NO trae tag, es la forma abreviada de Ollama para
        `:latest`, así que solo matchea si ese tag específico está presente.
        """
        if ":" in configured:
            return configured in available
        return f"{configured}:latest" in available

    def ensure_model_available(self, *, timeout: float = 10.0) -> None:
        """Chequeo de arranque: server reachable + modelo configurado presente.

        No imprime ni aborta el proceso — solo levanta las excepciones
        tipadas de arriba; `main.py` decide el mensaje y el `SystemExit`
        (ver diseño, ADR-1/ADR-2).
        """
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
            names = [model["name"] for model in response.json()["models"]]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise OllamaUnreachableError(
                f"no se pudo consultar {self._base_url}/api/tags: {exc}"
            ) from exc

        if not self._model_matches(self._model, names):
            raise OllamaModelMissingError(self._model, available=names)

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
        return self._backfill_structure(user_text, self.parse_llm_output(raw))

    def extract_structure(self, user_text: str) -> dict:
        """Segunda pasada enfocada: extrae SOLO la estructura (formula/
        red_cristalina) del pedido, sin clasificar intención.

        Necesaria porque cuando el mensaje trae OTRO eje de extracción —el
        tipo de cálculo en `preparar_calculo`, la cara/capas/supercelda en
        una losa— el modelo se lleva la atención a ese eje y suelta el
        material; una llamada sin ese framing lo recupera de forma
        confiable (ver `_STRUCT_PROMPT`). Medido en qwen2.5:7b, gemma4:12b
        y qwen2.5-coder:14b: los tres pierden `formula` en los pedidos de
        losa, así que es del prompt, no del modelo.

        Devuelve solo las claves de estructura y descarta los vacíos: el
        modelo emite `""` (no `null`) para lo que no encontró, y un
        `red_cristalina=""` que llegue a `StructureRequest` no queda
        ignorado sino que REBOTA (`_v_crystal`), cambiando una repregunta
        clara por un error de validación. `{}` si Ollama no respondió o la
        salida quedó fuera de schema."""
        raw = self._chat(_STRUCT_PROMPT, user_text, self._params_schema)
        if raw is None:
            return {}
        try:
            params = RouterParams.model_validate_json(raw)
        except ValidationError:
            logger.warning("Extracción de estructura fuera de schema: %r", (raw or "")[:200])
            return {}
        extracted = params.model_dump(exclude_none=True)
        return {
            k: v for k in _STRUCTURE_KEYS
            if (v := extracted.get(k)) not in (None, "", [], {})
        }

    def _backfill_structure(self, user_text: str, plan: Plan) -> Plan:
        """Recupera formula/red_cristalina vía la segunda pasada de
        `extract_structure`.

        Se aplica cuando el plan tiene EXACTAMENTE UN paso que necesita
        material (`_STRUCTURE_INTENTS`) y ese paso no lo trae: ahí el
        material del texto es inequívoco, sin importar si además hay pasos
        que no lo necesitan (p. ej. el `crear_directorio` que acompaña al
        cálculo en cada instrucción que emite el descompositor). Con dos o
        más el material deja de ser inequívoco y el caso lo cubre el
        descompositor (capa de aplicación), que primero parte el pedido
        para que cada instrucción tenga un solo material antes de
        re-rutear — cada instrucción vuelve por acá y se completa.

        Fail-open: si la segunda pasada no devuelve nada se conserva el plan
        original, y el handler repregunta el material faltante. Lo ya
        extraído por el router gana sobre el backfill."""
        targets = [s for s in plan.steps if s.action in _STRUCTURE_INTENTS]
        if len(targets) != 1 or targets[0].parametros.get("formula"):
            return plan
        structure = self.extract_structure(user_text)
        if not structure:
            return plan
        target = targets[0]
        steps = [
            PlanStep(action=s.action, parametros={**structure, **s.parametros})
            if s is target else s
            for s in plan.steps
        ]
        return Plan(steps=steps)

    def decompose(self, user_text: str) -> list[str]:
        """Divide un pedido largo en instrucciones simples auto-contenidas.

        Etapa intermedia para pedidos compuestos que exceden el techo de
        extracción del modelo (medido: en mensajes largos emite la
        estructura del plan bien pero omite formula/red_cristalina de los
        pasos de cálculo). Cada instrucción resultante se rutea por
        separado con `route()`, el camino corto que el modelo resuelve de
        forma confiable. Lista vacía = no se pudo descomponer (el llamador
        decide el fallback)."""
        raw = self._chat(
            _DECOMPOSE_PROMPT, user_text, _Decomposition.model_json_schema()
        )
        if raw is None:
            return []
        try:
            parsed = _Decomposition.model_validate_json(raw)
        except ValidationError:
            logger.warning("decompose devolvió JSON inválido; sin descomposición")
            return []
        return [i.strip() for i in parsed.instrucciones if i.strip()]

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

    def extract_edit(self, plan_context: str, user_text: str) -> tuple[Optional[int], dict]:
        """Cambio sobre un plan de VARIOS pasos: además del delta de
        parámetros, intenta identificar a qué paso (1-based) se refiere
        el mensaje, semántica o explícitamente ('paso N'). `(None, {})`
        o `target_index=None` si no hay nada que extraer o el LLM no
        tiene confianza — nunca se adivina un paso (ver diseño §3.2)."""
        system_prompt = _EDIT_TARGET_PROMPT_TEMPLATE.format(plan_context=plan_context)
        raw = self._chat(system_prompt, user_text, self._edit_decision_schema)
        decision = self.parse_edit_output(raw)
        return decision.target_index, decision.parametros.model_dump(exclude_none=True)

    @staticmethod
    def parse_edit_output(raw: Optional[str]) -> EditDecision:
        """Validación final de `extract_edit`, fail-closed: cualquier
        salida fuera de schema (o `raw is None`, p. ej. Ollama caído) se
        trata como "sin confianza" — `EditDecision()` por defecto ya es
        el resultado más seguro (`target_index=None`, sin delta), así
        que el llamador nunca fusiona ni ejecuta a ciegas."""
        try:
            return EditDecision.model_validate_json(raw or "")
        except ValidationError:
            logger.warning("Extracción de edición fuera de schema: %r", (raw or "")[:200])
            return EditDecision()

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
