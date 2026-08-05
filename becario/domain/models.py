"""Modelos de dominio de B.E.C.A.R.I.O.

Toda la validación/sanitización de parámetros vive acá, en el centro de la
arquitectura. Las capas externas (Telegram, SSH, LLM) nunca construyen
comandos a partir de strings sin pasar por estos modelos.
"""
from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .vasp_tags import TAGS_CON_CAMPO_PROPIO, validar_tags_pedidos

if TYPE_CHECKING:  # solo para anotar; el dominio no importa ASE en runtime
    from ase import Atoms

# ---------------------------------------------------------------------------
# Intenciones
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    """Acciones que el enrutador LLM puede decidir."""

    MODIFY_STRUCTURE = "modificar_estructura"
    PREPARE_CALC = "preparar_calculo"
    SUBMIT_SLURM = "enviar_slurm"
    QUERY_DB = "consultar_db"
    QUERY_RESULTS = "consultar_resultados"
    CHECK_STATUS = "revisar_estado"
    CANCEL_JOB = "cancelar_calculo"
    CREATE_DIR = "crear_directorio"
    LIST_FILES = "listar_archivos"
    VIEW_FILE = "ver_archivo"
    UNKNOWN = "error"

    @classmethod
    def destructive(cls) -> frozenset["Intent"]:
        """Intenciones que requieren confirmación humana."""
        return frozenset({cls.SUBMIT_SLURM, cls.CANCEL_JOB})


# ---------------------------------------------------------------------------
# Value objects validados (Pydantic)
# ---------------------------------------------------------------------------

# Nota: se usa \Z (fin absoluto) y no $ porque en Python `$` también matchea
# antes de un '\n' final — un newline colado permitiría inyectar líneas.
_JOB_NAME_RE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")
_PARTITION_RE = re.compile(r"\A[A-Za-z0-9_-]{1,32}\Z")
_JOB_ID_RE = re.compile(r"\A[0-9]{1,12}(_[0-9]{1,6})?\Z")  # soporta arrays 123_4
_TIME_RE = re.compile(r"\A(\d{1,3}-)?\d{1,3}:\d{2}:\d{2}\Z")  # [D-]HH:MM:SS
_SCRIPT_PATH_RE = re.compile(r"\A/[A-Za-z0-9_\-./]+\Z")
_FORMULA_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9]{0,15}\Z")  # Si, NaCl, H2O, TiO2
_UNIX_USER_RE = re.compile(r"\A[a-z_][a-z0-9_-]{0,31}\Z")
# Nombre de archivo suelto (CONTCAR, OSZICAR, run.sh): sin separadores de
# ruta ni '..', para que al resolverlo contra el dir de una corrida no se
# pueda escapar de ese directorio (path traversal).
_REMOTE_FILENAME_RE = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")

# Símbolos de la tabla periódica (los 118 elementos conocidos). Pura data
# de Python: nada de dependencias pesadas (p. ej. `ase`) solo para validar
# que una fórmula tiene sentido químico.
_ELEMENT_SYMBOLS: frozenset[str] = frozenset({
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
})

# Token de fórmula: el símbolo del elemento (mayúscula + minúscula
# opcional, capturado en el grupo 1) seguido de un subíndice numérico
# opcional que se descarta (Zr, H2, Na -> Zr, H, Na).
_ELEMENT_TOKEN_RE = re.compile(r"([A-Z][a-z]?)[0-9]*")


def is_plausible_formula(formula: str) -> bool:
    """¿`formula` parece una fórmula química real (y no, p. ej., un
    identificador mal resuelto como 'ultimo_calculo')?

    Dos filtros en cascada:
    1. `_FORMULA_RE`: forma sintáctica básica (alfanumérico, sin '_',
       arranca con letra, largo razonable).
    2. Tokenización en símbolos de elemento: la fórmula completa debe
       poder leerse como una secuencia de símbolos válidos de la tabla
       periódica (p. ej. 'ultimocalculo' pasa el filtro 1 pero no
       tokeniza en elementos reales).
    """
    if not _FORMULA_RE.match(formula):
        return False
    pos = 0
    while pos < len(formula):
        match = _ELEMENT_TOKEN_RE.match(formula, pos)
        if not match or match.group(1) not in _ELEMENT_SYMBOLS:
            return False
        pos = match.end()
    return True


# Campos numéricos donde un valor inventado NO falla: entra al INCAR o al
# KPOINTS, la corrida sale y el resultado parece válido. `tiempo_limite` y
# `miller` quedan afuera a propósito: el primero se re-formatea ("2hs" ->
# "02:00:00", los dígitos no sobreviven) y el segundo se escribe pegado
# ("(001)"), así que sus cifras aparecen igual y el chequeo no distinguiría
# nada.
_CAMPOS_CON_NUMERO_LITERAL = frozenset({
    "encut", "encut_min", "encut_max", "encut_paso", "parametro_red",
    "capas", "nodos", "vacio", "nbands", "puntos_k", "supercelda",
})


def _numeros_de(valor) -> list[str]:
    """Las cifras de un valor, como texto, para buscarlas en el mensaje."""
    if isinstance(valor, bool) or valor is None:
        return []
    if isinstance(valor, (int, float)):
        crudo = f"{valor:g}" if isinstance(valor, float) else str(valor)
        return [crudo]
    if isinstance(valor, (list, tuple)):
        return [n for v in valor for n in _numeros_de(v)]
    return []


def descartar_numeros_inventados(params: dict, texto: str) -> tuple[dict, list[str]]:
    """Saca los campos numéricos cuyo número NO está escrito en el mensaje.

    Devuelve `(params_limpios, campos_descartados)`.

    Frontera anti-alucinación, de la misma familia que `is_plausible_formula`:
    no depende de que el modelo se porte bien. Medido dos veces en producción
    —un `parametro_red=5.63` inventado al responder una pregunta que no traía
    ningún número, y un `puntos_k=[1,1,1]` agregado a un pedido que solo
    hablaba de ENCUT, nodos y tiempo— y las dos veces el número habría entrado
    a la corrida sin que nada fallara. Un k-mesh 1x1x1 es solo el punto Γ:
    corre igual y da otra física.

    Solo se mira lo que el usuario ESCRIBIÓ. Si dijo el número, pasa; si no
    aparece por ningún lado, no hay de dónde lo haya sacado.
    """
    limpio, descartados = {}, []
    for campo, valor in (params or {}).items():
        numeros = _numeros_de(valor) if campo in _CAMPOS_CON_NUMERO_LITERAL else []
        if numeros and any(n not in (texto or "") for n in numeros):
            descartados.append(campo)
            continue
        limpio[campo] = valor
    return limpio, descartados


def elements_of(formula: str) -> list[str]:
    """Símbolos de elemento (en orden de aparición, sin repetir) de una
    fórmula. 'Fe2O3' -> ['Fe', 'O']; 'W' -> ['W']. Sirve para distinguir un
    elemento simple (ASE) de un compuesto (Materials Project)."""
    out: list[str] = []
    pos = 0
    while pos < len(formula):
        match = _ELEMENT_TOKEN_RE.match(formula, pos)
        if not match:
            break
        symbol = match.group(1)
        if symbol not in out:
            out.append(symbol)
        pos = match.end()
    return out


# Redes que `ase.build.bulk` sabe construir. Se listan acá, en el dominio, en
# vez de importarlas de ASE: el dominio no depende de infraestructura. El test
# `test_ase_crystals_matches_ase` chequea que la lista no se desincronice.
ASE_CRYSTALS: frozenset[str] = frozenset({
    "sc", "fcc", "bcc", "tetragonal", "bct", "hcp", "rhombohedral",
    "orthorhombic", "mcl", "diamond", "zincblende", "rocksalt",
    "cesiumchloride", "fluorite", "wurtzite",
})

# El usuario dicta en castellano y ASE espera el nombre en inglés. Sin este
# mapeo, "fluorita" pasa la validación de forma y explota recién adentro de
# ASE, con un error que no dice qué corregir.
_CRYSTAL_ALIASES: dict[str, str] = {
    "fluorita": "fluorite",
    "diamante": "diamond",
    "wurtzita": "wurtzite",
    "zincblenda": "zincblende",
    "blenda": "zincblende",
    "salderoca": "rocksalt",
    "salgema": "rocksalt",
    "halita": "rocksalt",
    "clorurodecesio": "cesiumchloride",
    "cubicasimple": "sc",
    "cubicacentradaenelcuerpo": "bcc",
    "cubicacentradaenlascaras": "fcc",
    "hexagonalcompacta": "hcp",
    "romboedrica": "rhombohedral",
    "ortorrombica": "orthorhombic",
    "monoclinica": "mcl",
}


# Los 7 sistemas cristalinos, con el valor EXACTO que devuelve Materials
# Project (`SummaryDoc.symmetry.crystal_system`). Son otra cosa que
# `ASE_CRYSTALS`: un prototipo de ASE ('fluorite') dice cómo se decoran los
# sitios; un sistema cristalino ('Tetragonal') nombra una FASE. Un compuesto
# con polimorfos —ZrO2 es monoclínica a ambiente, tetragonal >1170 °C y
# cúbica >2370 °C— se pide por fase, que es como se lo nombra en materiales.
CRYSTAL_SYSTEMS: dict[str, str] = {
    "triclinic": "Triclinic",
    "monoclinic": "Monoclinic",
    "orthorhombic": "Orthorhombic",
    "tetragonal": "Tetragonal",
    "trigonal": "Trigonal",
    "hexagonal": "Hexagonal",
    "cubic": "Cubic",
}

_SYSTEM_ALIASES: dict[str, str] = {
    "triclinica": "triclinic",
    "monoclinica": "monoclinic",
    "ortorrombica": "orthorhombic",
    "rombica": "orthorhombic",
    "trigonal": "trigonal",
    "romboedrica": "trigonal",
    "cubica": "cubic",
    "hexagonal": "hexagonal",
    # 'tetragonal' y 'orthorhombic' ya coinciden con la clave en inglés.
}

# Nombre en castellano de cada fase, para hablarle al usuario.
CRYSTAL_SYSTEM_ES: dict[str, str] = {
    "Triclinic": "triclínica",
    "Monoclinic": "monoclínica",
    "Orthorhombic": "ortorrómbica",
    "Tetragonal": "tetragonal",
    "Trigonal": "trigonal",
    "Hexagonal": "hexagonal",
    "Cubic": "cúbica",
}


def _plain(raw: str) -> str:
    """Minúsculas, sin acentos y sin separadores: la forma en la que se
    comparan los nombres que llegan dictados o escritos por un LLM."""
    text = unicodedata.normalize("NFKD", raw.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[\s_-]+", "", text)


def normalize_crystal_system(raw: str) -> Optional[str]:
    """Sistema cristalino en el valor que usa Materials Project, o `None`.

    'monoclínica' -> 'Monoclinic'. Se acepta el castellano porque es como se
    nombra la fase en el pedido ("la ZrO2 tetragonal"), no en inglés.
    """
    plain = _plain(raw)
    key = _SYSTEM_ALIASES.get(plain, plain)
    return CRYSTAL_SYSTEMS.get(key)


def normalize_crystal(raw: str) -> Optional[str]:
    """Nombre de red en la forma que entiende ASE, o `None` si no existe.

    Tolera acentos, mayúsculas y separadores ('Sal de Roca' -> 'rocksalt')
    porque el texto viene de un dictado por voz o de un LLM, no de un menú.
    """
    plain = _plain(raw)
    if plain in ASE_CRYSTALS:
        return plain
    return _CRYSTAL_ALIASES.get(plain)


def resolve_crystal(v: Optional[str], formula: str) -> Optional[str]:
    """Política única de red cristalina, compartida por `StructureRequest` y
    `VaspCalcRequest` para que no diverjan.

    El campo carga DOS sentidos y cuál aplica depende del material:

    - **Elemento** -> prototipo de ASE. 'monoclínica' es `mcl`, una red de
      Bravais de un átomo, que es exactamente lo que hace falta para armar
      un elemento con `ase.build.bulk`.
    - **Compuesto** -> sistema cristalino, o sea una FASE ('Monoclinic').
      Acá `mcl` no sirve para nada: pide un solo átomo y un compuesto tiene
      varios. Un compuesto con polimorfos se resuelve por Materials Project,
      que indexa las fases por sistema.

    Por eso la resolución mira la fórmula en vez de decidirse sola: sin ese
    contexto, 'monoclínica' se iría siempre al prototipo de ASE y la fase
    quedaría muda. Los nombres que solo existen en un vocabulario
    ('fluorita', 'fcc') no dependen del contexto y se resuelven igual.

    Rechazar lo que no está en ninguno de los dos es fail-closed a propósito:
    el camino largo es generar el INCAR, subirlo y que el cálculo muera en el
    cluster por una red mal escrita."""
    if v is None:
        return None
    es_compuesto = len(elements_of(formula)) > 1
    orden = (
        (normalize_crystal_system, normalize_crystal)
        if es_compuesto
        else (normalize_crystal, normalize_crystal_system)
    )
    for resolver in orden:
        resolved = resolver(v)
        if resolved is not None:
            return resolved
    conocidas = ", ".join(sorted(ASE_CRYSTALS | set(CRYSTAL_SYSTEMS.values())))
    raise ValueError(f"red cristalina inválida: {v!r} (conocidas: {conocidas})")


def needs_explicit_lattice(
    formula: str, crystal: Optional[str], lattice_a: Optional[float]
) -> bool:
    """¿Hay que pedirle al usuario la red y el parámetro antes de construir?

    ASE trae la estructura de referencia de cada ELEMENTO, así que 'W' o 'Si'
    se arman solos. Un compuesto no: `bulk('ZrO2')` falla con "no suitable
    reference data" salvo que se le den red y parámetro. Preguntarlos es lo
    mismo que ya se hace con la cara de una losa — el parámetro de red no se
    adivina, y uno inventado da una estructura creíble y equivocada.
    """
    return len(elements_of(formula)) > 1 and not (crystal and lattice_a)


def _validate_remote_dir_path(v: str) -> str:
    """Política única para directorios remotos: ruta absoluta, sin
    metacaracteres de shell y sin '..'. La comparten `RemoteDirRequest`
    y `StructureRequest.remote_dest_dir` para que no diverjan."""
    if not _SCRIPT_PATH_RE.match(v) or ".." in v:
        raise ValueError(f"directorio remoto inválido: {v!r}")
    return v


class SlurmJobRequest(BaseModel):
    """Parámetros validados para un envío sbatch."""

    job_name: str = Field(default="becario_job")
    partition: str = Field(default="default")
    nodes: int = Field(default=1, ge=1, le=64)
    time_limit: str = Field(default="01:00:00")
    script_path: str

    @field_validator("job_name")
    @classmethod
    def _v_job_name(cls, v: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_-]", "_", v)[:64] or "becario_job"
        if not _JOB_NAME_RE.match(clean):
            raise ValueError(f"job_name inválido: {v!r}")
        return clean

    @field_validator("partition")
    @classmethod
    def _v_partition(cls, v: str) -> str:
        if not _PARTITION_RE.match(v):
            raise ValueError(f"partition inválida: {v!r}")
        return v

    @field_validator("time_limit")
    @classmethod
    def _v_time(cls, v: str) -> str:
        if not _TIME_RE.match(v):
            raise ValueError(f"time_limit inválido: {v!r} (formato HH:MM:SS o D-HH:MM:SS)")
        return v

    @field_validator("script_path")
    @classmethod
    def _v_script(cls, v: str) -> str:
        if not _SCRIPT_PATH_RE.match(v):
            raise ValueError(f"script_path inválido: {v!r} (ruta absoluta, sin caracteres especiales)")
        if ".." in v:
            raise ValueError("script_path no puede contener '..'")
        return v

    def describe(self) -> str:
        return (
            f"nombre: {self.job_name}\n"
            f"partición: {self.partition}\n"
            f"nodos: {self.nodes}\n"
            f"tiempo: {self.time_limit}\n"
            f"script: {self.script_path}"
        )


class JobId(BaseModel):
    """Identificador de trabajo Slurm validado."""

    value: str

    @field_validator("value")
    @classmethod
    def _v(cls, v: str) -> str:
        v = str(v).strip()
        if not _JOB_ID_RE.match(v):
            raise ValueError(f"job_id inválido: {v!r}")
        return v

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class RemoteDirRequest(BaseModel):
    """Directorio remoto validado a crear en el cluster.

    Misma política de rutas que `remote_dest_dir` en `StructureRequest`:
    absoluta, sin metacaracteres de shell y sin '..'.
    """

    path: str

    @field_validator("path")
    @classmethod
    def _v_path(cls, v: str) -> str:
        return _validate_remote_dir_path(v.strip())


class ListFilesRequest(BaseModel):
    """Directorio remoto validado a listar (solo lectura).

    Comparte la política de rutas de `RemoteDirRequest` vía
    `_validate_remote_dir_path`.
    """

    path: str

    @field_validator("path")
    @classmethod
    def _v_path(cls, v: str) -> str:
        return _validate_remote_dir_path(v.strip())


class ViewFileRequest(BaseModel):
    """Archivo remoto a mostrar (solo lectura). Dos formas mutuamente
    excluyentes de identificarlo, y hay que dar exactamente una:

    - `path`: ruta absoluta completa al archivo (misma política de rutas
      que `ListFilesRequest`).
    - `filename`: nombre suelto (p. ej. CONTCAR) que el servicio resuelve
      contra el directorio de la última corrida. Nunca lleva separadores
      de ruta ni es '.'/'..', así que el nombre resuelto no puede escapar
      del directorio de la corrida.
    """

    path: Optional[str] = None
    filename: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _v_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_remote_dir_path(v.strip())

    @field_validator("filename")
    @classmethod
    def _v_filename(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if v in (".", "..") or not _REMOTE_FILENAME_RE.match(v):
            raise ValueError(f"nombre de archivo inválido: {v!r}")
        return v

    @model_validator(mode="after")
    def _exactly_one(self) -> "ViewFileRequest":
        # Corre siempre (a diferencia de los field_validator, que se saltean
        # los defaults None): exige exactamente una de las dos formas.
        if bool(self.path) == bool(self.filename):
            raise ValueError("hay que dar 'path' o 'filename', y solo uno")
        return self


class HistoryFilter(BaseModel):
    """Filtro de búsqueda para el historial (parametrizado, nunca interpolado).

    `owner_id` NUNCA lo completa el LLM: lo asigna el servicio de aplicación
    a partir del remitente autenticado, para que nadie pueda pedirle al LLM
    que le muestre el historial de otra persona.
    """

    job_id: Optional[str] = None
    name_contains: Optional[str] = None
    owner_id: Optional[int] = None
    limit: int = Field(default=5, ge=1, le=50)

    @field_validator("job_id")
    @classmethod
    def _v_jid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip()
        if not _JOB_ID_RE.match(v):
            raise ValueError(f"job_id inválido: {v!r}")
        return v

    @field_validator("name_contains")
    @classmethod
    def _v_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Se usa con query parametrizada; solo limitamos longitud.
        return v.strip()[:80] or None


# ---------------------------------------------------------------------------
# Identidad multiusuario
# ---------------------------------------------------------------------------


class ClusterIdentity(BaseModel):
    """La cuenta propia de un miembro del grupo en el cluster.

    No hay un rol admin: cada persona opera únicamente con su cuenta SSH,
    así que Slurm mismo impide que alguien toque los trabajos de otro
    (scancel/sacct fallan con permiso denegado fuera de la propia cuenta).
    """

    telegram_user_id: int
    ssh_user: str
    ssh_key_path: str
    display_name: str = ""
    ssh_host: Optional[str] = None  # None => usa el host global de Settings

    @field_validator("ssh_user")
    @classmethod
    def _v_ssh_user(cls, v: str) -> str:
        if not _UNIX_USER_RE.match(v):
            raise ValueError(f"ssh_user inválido: {v!r}")
        return v


# ---------------------------------------------------------------------------
# Estructuras atómicas (ASE)
# ---------------------------------------------------------------------------


class StructureKind(str, Enum):
    BULK = "bulk"
    MOLECULE = "molecule"
    SLAB = "slab"  # superficie: losa con vacío, orientada por índice de Miller


class Axis(str, Enum):
    """Eje cartesiano. Se usa para elegir dónde queda el vacío de una losa."""

    X = "x"
    Y = "y"
    Z = "z"

    @property
    def index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[self.value]


# Vacío por default de una losa, en Å. Baja del TIPO de estructura, no del
# campo: un bulk no lleva vacío, y una losa sin vacío no es una superficie
# (las imágenes periódicas se tocarían a través del borde de la celda).
SLAB_DEFAULT_VACUUM = 15.0

# Capas por default de una losa. Es un punto de PARTIDA, no un valor
# convergido: el espesor hay que convergerlo por sistema, así que el
# resultado siempre reporta cuántas capas se usaron.
SLAB_DEFAULT_LAYERS = 5


def _validate_miller(
    v: Optional[tuple[int, int, int]],
) -> Optional[tuple[int, int, int]]:
    """Índice de Miller: tres enteros, no todos cero, en un rango sano."""
    if v is None:
        return None
    if len(v) != 3:
        raise ValueError(f"índice de Miller inválido: {v!r} (necesita 3 enteros)")
    idx = tuple(int(x) for x in v)
    if all(x == 0 for x in idx):
        raise ValueError("(000) no define un plano")
    if any(abs(x) > 9 for x in idx):
        raise ValueError(f"índice de Miller fuera de rango: {idx!r}")
    return idx


def validate_slab_spec(
    kind: StructureKind,
    miller: Optional[tuple[int, int, int]],
    layers: Optional[int],
    vacuum: Optional[float],
    supercell: tuple[int, int, int],
) -> tuple[Optional[tuple[int, int, int]], Optional[int], Optional[float]]:
    """Reglas de losa compartidas por `StructureRequest` y `VaspCalcRequest`.

    Los dos modelos replican a propósito la parte estructural (ver el
    docstring de `VaspCalcRequest`), así que la regla vive UNA vez acá: si se
    duplicara, un pedido de losa por el camino de cálculo podría terminar
    generando inputs de bulk sin que nada avise.

    Devuelve la terna `(miller, layers, vacuum)` ya resuelta.
    """
    if kind is not StructureKind.SLAB:
        # No se ignoran en silencio: si vienen, el pedido está mal armado.
        if miller is not None or layers is not None:
            raise ValueError(
                "índice de Miller y capas solo aplican a una losa "
                "(tipo_estructura=slab)"
            )
        return None, None, vacuum

    if miller is None:
        # No se adivina una orientación: elegir la cara equivocada cambia la
        # física (terminación, polaridad, estados de superficie).
        raise ValueError(
            "una losa necesita índice de Miller (p. ej. 001, 110, 111)"
        )
    if supercell[2] != 1:
        raise ValueError(
            "la supercelda de una losa se repite solo en el plano: la tercera "
            "dimensión debe ser 1 (repetirla duplicaría la losa y su vacío)"
        )
    resolved_layers = SLAB_DEFAULT_LAYERS if layers is None else layers
    resolved_vacuum = SLAB_DEFAULT_VACUUM if vacuum is None else vacuum
    return miller, resolved_layers, resolved_vacuum


class CalcKind(str, Enum):
    """Tipos de cálculo VASP que BECARIO sabe preparar."""

    RELAX = "relajacion"  # ISIF=3: relaja también los parámetros de red
    STATIC = "estatico"
    ENCUT_SCAN = "convergencia_encut"
    DOS = "dos"  # densidad de estados: punto único, tetraedros, malla densa


class OutputFormat(str, Enum):
    VASP = "vasp"  # POSCAR
    CIF = "cif"
    XYZ = "xyz"


class StructureSource(str, Enum):
    """De dónde sale la estructura de un cálculo.

    `AUTO` deja que el ruteo decida (elemento -> ASE, compuesto -> MP);
    `ASE`/`MP` fuerzan una fuente. El default `AUTO` preserva el comportamiento
    actual, así el campo queda inerte si no se usa.
    """

    AUTO = "auto"
    ASE = "ase"
    MP = "mp"
    # Partir del CONTCAR de la última relajación propia de ese material
    # ("el bulk de ZrO2 relajado"). No es una fuente externa: es un
    # resultado anterior del propio usuario.
    RELAXED = "relajado"


class StructureRequest(BaseModel):
    """Pedido validado de construcción de una estructura atómica."""

    formula: str
    kind: StructureKind = StructureKind.BULK
    crystal: Optional[str] = None  # p. ej. diamond, fcc, rocksalt, zincblende
    lattice_a: Optional[float] = Field(default=None, gt=0.5, lt=50.0)  # Å
    supercell: tuple[int, int, int] = (1, 1, 1)
    vacuum: Optional[float] = Field(default=None, ge=0.0, le=60.0)  # Å
    # Losa (kind=SLAB): orientación de la cara que mira al vacío y espesor.
    # `vacuum_axis` es solo convención de ejes — el Miller define la cara, así
    # que pedir el vacío en otro eje devuelve la MISMA superficie acostada.
    miller: Optional[tuple[int, int, int]] = None
    layers: Optional[int] = Field(default=None, ge=1, le=40)
    vacuum_axis: Axis = Axis.Z
    output_format: OutputFormat = OutputFormat.VASP
    remote_dest_dir: Optional[str] = None  # si se define, se sube por SFTP
    # Materials Project: id explícito y/o fuente forzada. Inertes por default.
    mp_id: Optional[str] = None
    source: StructureSource = StructureSource.AUTO

    @field_validator("formula")
    @classmethod
    def _v_formula(cls, v: str) -> str:
        v = v.strip()
        if not is_plausible_formula(v):
            raise ValueError(f"fórmula inválida: {v!r}")
        return v

    @field_validator("mp_id")
    @classmethod
    def _v_mp_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not _MP_ID_RE.match(v):
            raise ValueError(f"mp_id inválido: {v!r} (formato mp-<número>)")
        return v

    @field_validator("crystal")
    @classmethod
    def _v_crystal(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # `formula` se declara antes que `crystal`, así que para cuando corre
        # esto ya está validada y disponible: es lo que decide si el valor es
        # un prototipo de ASE o una fase (ver `resolve_crystal`).
        return resolve_crystal(v, str(info.data.get("formula") or ""))

    @field_validator("supercell")
    @classmethod
    def _v_supercell(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(v) != 3 or any(not (1 <= n <= 10) for n in v):
            raise ValueError(f"supercelda inválida: {v!r} (cada dimensión entre 1 y 10)")
        return v

    @field_validator("remote_dest_dir")
    @classmethod
    def _v_remote(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_remote_dir_path(v)

    @field_validator("miller")
    @classmethod
    def _v_miller(cls, v):
        return _validate_miller(v)

    @model_validator(mode="after")
    def _v_slab(self) -> "StructureRequest":
        self.miller, self.layers, self.vacuum = validate_slab_spec(
            self.kind, self.miller, self.layers, self.vacuum, self.supercell
        )
        return self


@dataclass(frozen=True)
class StructureResult:
    """Estructura construida: archivo local + resumen físico."""

    local_path: str
    filename: str
    chemical_formula: str
    n_atoms: int
    cell_summary: str
    uploaded_to: Optional[str] = None
    # Resumen de losa. Se reporta SIEMPRE que haya: las capas y el vacío
    # pueden venir de un default, y un espesor sin converger cambia la
    # física — quien lo pidió tiene que ver con qué se armó.
    slab_summary: Optional[str] = None

    def describe(self) -> str:
        lines = [
            f"fórmula: {self.chemical_formula}",
            f"átomos: {self.n_atoms}",
            f"celda: {self.cell_summary}",
        ]
        if self.slab_summary:
            lines.append(f"losa: {self.slab_summary}")
        lines.append(f"archivo: {self.filename}")
        if self.uploaded_to:
            lines.append(f"subido a: {self.uploaded_to}")
        return "\n".join(lines)





# ---------------------------------------------------------------------------
# Cálculos VASP (inputs completos + workflows)
# ---------------------------------------------------------------------------


class VaspCalcRequest(BaseModel):
    """Pedido validado de un cálculo VASP completo (inputs + job).

    La parte estructural replica los campos de bulk de `StructureRequest`;
    la parte de job replica los defaults de `SlurmJobRequest`. El POTCAR no
    se describe acá: lo arma el servicio contra la biblioteca del cluster.
    """

    formula: str
    crystal: Optional[str] = None
    lattice_a: Optional[float] = Field(default=None, gt=0.5, lt=50.0)  # Å
    supercell: tuple[int, int, int] = (1, 1, 1)
    # Parte estructural de una losa, con las MISMAS reglas que
    # `StructureRequest` (`validate_slab_spec`). Sin esto, pedir "armá el slab
    # y mandalo a relajar" generaría inputs de bulk: el generador tiene su
    # propio camino de construcción y no ve el `StructureRequest`.
    kind: StructureKind = StructureKind.BULK
    miller: Optional[tuple[int, int, int]] = None
    layers: Optional[int] = Field(default=None, ge=1, le=40)
    vacuum: Optional[float] = Field(default=None, ge=0.0, le=60.0)  # Å
    vacuum_axis: Axis = Axis.Z
    calc_kind: CalcKind = CalcKind.STATIC
    encut: int = Field(default=520, ge=100, le=1500)  # eV
    kpoints: Optional[tuple[int, int, int]] = None  # None => grilla automática
    encut_values: Optional[list[int]] = None  # solo para ENCUT_SCAN
    # NSW: presupuesto de pasos iónicos. None => default por tipo de cálculo
    # (`_DEFAULT_NSW` en el generador: 180 al relajar, 0 en los tipos de un
    # solo punto). El TIPO MANDA sobre el pedido: en un estático o un barrido
    # de ENCUT se fuerza a 0 aunque acá venga otro valor — un cálculo de un
    # solo punto con pasos iónicos no es un estático, es una relajación.
    nsw: Optional[int] = Field(default=None, ge=0, le=10000)
    # ISIF: qué relaja la corrida de relajación. None => default por tipo de
    # cálculo (RELAX usa 3: iones + forma + volumen). Se puede pedir 2 para
    # relajar solo iones (típico en slabs/interfaces). Solo tiene efecto con
    # NSW>0, así que el generador lo emite únicamente en RELAX.
    isif: Optional[int] = Field(default=None, ge=0, le=7)
    # NBANDS: None => lo calcula el generador a partir del conteo de electrones
    # (tabla ZVAL) o, si no puede, lo deja a criterio de VASP. Se puede forzar
    # un número mayor cuando hacen falta bandas vacías (DOS, estados desocupados).
    nbands: Optional[int] = Field(default=None, ge=1, le=100000)
    # Tags del INCAR pedidos a mano, validados contra el vocabulario del
    # manual (`vasp_tags`). Es UN campo para los 169 tags en vez de un campo
    # por tag: el schema del router tiene presupuesto (ADR-0006) y agregar
    # uno por cada tag que alguien necesite lo agota en tres features.
    # Los que decide el tipo de cálculo y los que ya tienen campo propio se
    # rechazan acá, no se ignoran.
    incar_tags: dict[str, str] = Field(default_factory=dict)
    # Materials Project: id explícito (fuerza fuente MP por id) y/o fuente
    # forzada (auto = el ruteo decide elemento->ASE, compuesto->MP). Inertes
    # por default, así el cambio es aditivo.
    mp_id: Optional[str] = None
    source: StructureSource = StructureSource.AUTO
    partition: str = Field(default="default")
    nodes: int = Field(default=1, ge=1, le=64)
    time_limit: str = Field(default="01:00:00")

    @field_validator("formula")
    @classmethod
    def _v_formula(cls, v: str) -> str:
        v = v.strip()
        if not is_plausible_formula(v):
            raise ValueError(f"fórmula inválida: {v!r}")
        return v

    @field_validator("mp_id")
    @classmethod
    def _v_mp_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not _MP_ID_RE.match(v):
            raise ValueError(f"mp_id inválido: {v!r} (formato mp-<número>)")
        return v

    @field_validator("crystal")
    @classmethod
    def _v_crystal(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # `formula` se declara antes que `crystal`, así que para cuando corre
        # esto ya está validada y disponible: es lo que decide si el valor es
        # un prototipo de ASE o una fase (ver `resolve_crystal`).
        return resolve_crystal(v, str(info.data.get("formula") or ""))

    @field_validator("supercell")
    @classmethod
    def _v_supercell(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(v) != 3 or any(not (1 <= n <= 10) for n in v):
            raise ValueError(f"supercelda inválida: {v!r} (cada dimensión entre 1 y 10)")
        return v

    @field_validator("partition")
    @classmethod
    def _v_partition(cls, v: str) -> str:
        if not _PARTITION_RE.match(v):
            raise ValueError(f"partition inválida: {v!r}")
        return v

    @field_validator("time_limit")
    @classmethod
    def _v_time(cls, v: str) -> str:
        if not _TIME_RE.match(v):
            raise ValueError(f"time_limit inválido: {v!r} (formato HH:MM:SS o D-HH:MM:SS)")
        return v

    @field_validator("kpoints")
    @classmethod
    def _v_kpoints(cls, v: Optional[tuple[int, int, int]]) -> Optional[tuple[int, int, int]]:
        if v is None:
            return None
        if len(v) != 3 or any(not (1 <= n <= 40) for n in v):
            raise ValueError(f"puntos k inválidos: {v!r} (cada dimensión entre 1 y 40)")
        return v

    @field_validator("incar_tags")
    @classmethod
    def _v_incar_tags(cls, v: dict, info: ValidationInfo) -> dict[str, str]:
        # `encut` y `nbands` se declaran antes, así que acá ya están: se
        # pasan para distinguir un tag que CONTRADICE al campo de uno que
        # repite el mismo número (ver `validar_tags_pedidos`).
        propios = {k: info.data.get(k.lower()) for k in TAGS_CON_CAMPO_PROPIO}
        return validar_tags_pedidos(v, propios)

    @field_validator("encut_values")
    @classmethod
    def _v_encut_values(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is None:
            return None
        values = sorted(set(int(x) for x in v))
        if not (2 <= len(values) <= 15):
            raise ValueError("el barrido de ENCUT necesita entre 2 y 15 valores distintos")
        if any(not (100 <= x <= 1500) for x in values):
            raise ValueError("cada ENCUT del barrido debe estar entre 100 y 1500 eV")
        return values

    @field_validator("miller")
    @classmethod
    def _v_miller(cls, v):
        return _validate_miller(v)

    @model_validator(mode="after")
    def _v_slab(self) -> "VaspCalcRequest":
        if self.kind is StructureKind.MOLECULE:
            raise ValueError(
                "los cálculos VASP de BECARIO son de bulk o de losa, no de "
                "moléculas aisladas"
            )
        if self.kind is not StructureKind.SLAB and self.vacuum is not None:
            # El generador de inputs solo aplica vacío al cortar una losa; en
            # un bulk lo descartaría en silencio. Se rechaza en vez de mentir.
            raise ValueError(
                "el vacío solo aplica a una losa (tipo_estructura=slab)"
            )
        self.miller, self.layers, self.vacuum = validate_slab_spec(
            self.kind, self.miller, self.layers, self.vacuum, self.supercell
        )
        return self

    @model_validator(mode="after")
    def _v_nsw_vs_kind(self) -> "VaspCalcRequest":
        """Una relajación con NSW=0 no relaja nada: el directorio de la
        corrida diría `relajacion` y sería un cálculo de un solo punto. El
        cero se pide eligiendo un cálculo estático, no vaciándole los pasos
        iónicos a una relajación. (El caso espejo — pedir pasos iónicos en un
        estático — no se rechaza: lo fuerza a 0 el generador, porque ahí el
        tipo de cálculo manda.)"""
        if self.calc_kind is CalcKind.RELAX and self.nsw == 0:
            raise ValueError(
                "una relajación necesita NSW>=1; para un solo punto pedí un "
                "cálculo estático"
            )
        return self

    @staticmethod
    def default_encut_values() -> list[int]:
        """Barrido razonable si el usuario no especifica rango."""
        return list(range(300, 651, 50))

    def scan_values(self) -> list[int]:
        return self.encut_values or self.default_encut_values()


@dataclass(frozen=True)
class CalcDirResult:
    """Directorio de corrida generado localmente, listo para subir.

    `elements` preserva el orden de especies del POSCAR: es el orden en el
    que hay que concatenar los POTCAR."""

    local_dir: str
    run_name: str
    files: list[str]  # rutas relativas a local_dir
    elements: list[str]
    chemical_formula: str
    n_atoms: int
    cell_summary: str
    kpoints: tuple[int, int, int]
    calc_kind: CalcKind
    encut_values: list[int]  # un solo valor si no es barrido

    def describe(self) -> str:
        encut_txt = (
            f"barrido ENCUT: {self.encut_values[0]}–{self.encut_values[-1]} eV "
            f"({len(self.encut_values)} puntos)"
            if self.calc_kind is CalcKind.ENCUT_SCAN
            else f"ENCUT: {self.encut_values[0]} eV"
        )
        kx, ky, kz = self.kpoints
        return "\n".join(
            [
                f"fórmula: {self.chemical_formula}",
                f"átomos: {self.n_atoms}",
                f"celda: {self.cell_summary}",
                encut_txt,
                f"k-points: {kx}×{ky}×{kz} (Γ-centrado)",
            ]
        )


@dataclass(frozen=True)
class RoutedRequest:
    """Salida del enrutador: intención + parámetros crudos del LLM."""

    intent: Intent
    params: dict


# ---------------------------------------------------------------------------
# Fuente de estructura cristalina (ASE local vs Materials Project)
# ---------------------------------------------------------------------------

# Id de material de Materials Project: 'mp-' seguido de dígitos (mp-19770).
_MP_ID_RE = re.compile(r"\Amp-\d+\Z")


class StructureResolutionReason(str, Enum):
    """Por qué falló resolver una estructura, para mapear a un ⚠️ claro."""

    NETWORK = "network"      # no se pudo llegar a Materials Project
    API = "api"              # MP respondió con error
    NO_MATCH = "no_match"    # la consulta no devolvió estructura usable


class StructureResolutionError(RuntimeError):
    """Error de resolución de estructura con una causa clasificada.

    Lo lanza el adaptador (infraestructura) y lo mapea la capa de aplicación
    a una respuesta al usuario, sin que el dominio toque excepciones crudas
    de pymatgen/httpx."""

    def __init__(self, reason: StructureResolutionReason, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason.value)


class StructureQuery(BaseModel):
    """Pedido neutral de estructura que la capa de aplicación arma a partir
    del request del usuario y le pasa al `StructureProvider`.

    El puerto no sabe de Telegram ni de Reply: solo recibe qué buscar.
    - `elements`: sistema químico (chemsys), p. ej. ('Fe', 'O').
    - `formula`: estequiometría explícita si el usuario la dio (Fe2O3).
    - `mp_id`: id explícito de Materials Project (bypassea la selección).
    - `qualifier`: elemento para filtrar el chemsys (O para óxidos).
    Hay que dar al menos una forma de identificar el material.
    """

    elements: tuple[str, ...] = ()
    formula: Optional[str] = None
    mp_id: Optional[str] = None
    qualifier: Optional[str] = None
    # Fase pedida, en el valor de MP ('Tetragonal'). Sin esto, resolver un
    # compuesto con polimorfos devuelve SIEMPRE el más estable, que para
    # ZrO2 es la monoclínica — correcto a temperatura ambiente y equivocado
    # si el usuario quería la tetragonal.
    crystal_system: Optional[str] = None

    @field_validator("crystal_system")
    @classmethod
    def _v_crystal_system(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        system = normalize_crystal_system(v)
        if system is None:
            conocidos = ", ".join(CRYSTAL_SYSTEMS.values())
            raise ValueError(f"sistema cristalino inválido: {v!r} ({conocidos})")
        return system

    @field_validator("mp_id")
    @classmethod
    def _v_mp_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not _MP_ID_RE.match(v):
            raise ValueError(f"mp_id inválido: {v!r} (formato mp-<número>)")
        return v

    @field_validator("formula")
    @classmethod
    def _v_formula(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not is_plausible_formula(v):
            raise ValueError(f"fórmula inválida: {v!r}")
        return v

    @model_validator(mode="after")
    def _at_least_one(self) -> "StructureQuery":
        if not (self.elements or self.formula or self.mp_id):
            raise ValueError("StructureQuery necesita elements, formula o mp_id")
        return self


@dataclass(frozen=True)
class StructureAlternative:
    """Un candidato de Materials Project que NO se eligió, ofrecido al usuario
    para que decida cuando el nombre era ambiguo (p. ej. 'óxido de hierro')."""

    mp_id: str
    formula: str
    energy_above_hull: float
    # Fase del candidato ('Tetragonal'). Es lo que permite ofrecer la
    # elección en el idioma en que se nombra la fase, en vez de listar ids
    # de MP que no le dicen nada a nadie.
    crystal_system: str = ""


@dataclass(frozen=True)
class StructureResolution:
    """Estructura resuelta por un `StructureProvider`: los átomos elegidos
    más metadatos neutrales.

    `atoms` es un `ase.Atoms` (el mismo tipo que ya consumen los generadores):
    el adaptador convierte desde pymatgen internamente y NUNCA filtra tipos de
    pymatgen a través del puerto. `alternatives` viaja como metadato plano para
    satisfacer la política de nombre ambiguo sin acoplar el dominio a MP."""

    atoms: "Atoms"
    mp_id: str
    formula: str
    spacegroup: str
    # Fase de la estructura elegida ('Tetragonal'). Separado del grupo
    # espacial porque es lo que el usuario nombra y compara entre polimorfos;
    # 'P4_2/nmc' es exacto y no le dice nada a nadie en una conversación.
    crystal_system: str = ""
    # `None` cuando la estructura se pidió por su mp-id: `get_structure_by_
    # material_id` no trae el summary, así que el hull es desconocido. No se
    # inventa 0.0 — la nota humana omite la afirmación de estabilidad si es None.
    energy_above_hull: Optional[float] = None
    alternatives: tuple[StructureAlternative, ...] = ()
    # TODOS los sistemas cristalinos que ese material tiene en la fuente.
    # Separado de `alternatives` porque responden preguntas distintas:
    # `alternatives` es "¿no querrías otro material?" (ordenadas por
    # estabilidad y recortadas), y esto es "¿qué fases hay de ESTE?". Medido
    # contra MP: ZrO2 tiene 20 estructuras en 5 sistemas y la cúbica —la de
    # tipo fluorita— queda 16ª por energía, o sea afuera del recorte. Listar
    # las fases desde las alternativas la habría escondido.
    phases: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Planes multi-paso (composición de intenciones)
# ---------------------------------------------------------------------------

# El cap ya no acota el blast-radius del plan directo (eso lo hace ahora la
# capa de aplicación: nada que supere la composición chica se auto-ejecuta,
# va a confirmación completa de batch — ver ADR-0007). Queda como tope
# estructural: el descompositor emite <=5 instrucciones y cada una expande a
# <=2 pasos, así que 11 acota una descomposición desbocada sin recortar un
# batch legítimo. El router directo sigue capado en 5 (RouterDecision).
_MAX_PLAN_STEPS = 11
# Umbral de auto-ejecución: un plan compuesto de hasta esta cantidad de pasos
# con a lo sumo un cálculo se materializa en el acto (camino de ADR-0006);
# por encima, o con varios cálculos, va a confirmación de batch (ADR-0007).
_MAX_AUTOMATERIALIZE_STEPS = 5


# Intenciones cuya identidad depende del material: dos de estas solo son
# "el mismo paso dicho dos veces" si ambas dicen SOBRE QUÉ trabajan. Las
# losas entran por MODIFY_STRUCTURE y también llevan `formula`.
_MATERIAL_INTENTS = frozenset({Intent.PREPARE_CALC, Intent.MODIFY_STRUCTURE})


def _sin_material(step: "PlanStep") -> bool:
    """¿Este paso se parece a su vecino solo porque le falta el material?"""
    return step.action in _MATERIAL_INTENTS and not step.parametros.get("formula")


class PlanStep(BaseModel):
    """Un paso del plan: una intención + sus parámetros crudos del LLM.

    `parametros` queda como `dict` (no como el `RouterParams` de
    infraestructura) para que el dominio no dependa de la capa de
    infraestructura — el mismo patrón que ya usa `RoutedRequest.params`.
    La validación fuerte de cada paso ocurre después, contra su modelo de
    request específico (`SlurmJobRequest`, `VaspCalcRequest`, etc.).
    """

    action: Intent
    parametros: dict = Field(default_factory=dict)


class Plan(BaseModel):
    """Plan ordenado de pasos, fail-closed en su forma.

    A lo sumo un paso destructivo (`Intent.destructive()`) y, si existe,
    debe ser el último — así la confirmación humana sigue gatillando
    únicamente sobre la cola irreversible del plan (ver ADR-0006).

    Los pasos consecutivos idénticos se colapsan antes de esa regla (ver
    `_v_collapse_stutter`), y el orden de los dos validadores importa: un
    `enviar_slurm` tartamudeado son dos destructivos, que la regla de
    abajo rechazaría entero. Colapsarlo primero lo deja en el único envío
    que el usuario pidió.
    """

    steps: list[PlanStep] = Field(min_length=1, max_length=_MAX_PLAN_STEPS)

    @field_validator("steps")
    @classmethod
    def _v_collapse_stutter(cls, steps: list[PlanStep]) -> list[PlanStep]:
        """Colapsa pasos consecutivos IDÉNTICOS Y COMPLETOS: el modelo
        tartamudea.

        Medido con `qwen2.5-coder:14b` sobre "relajá el bulk de Zr hcp":
        devuelve `[preparar_calculo, preparar_calculo]` con los mismos
        parámetros, unánime en 3 de 3 intentos. Dos pasos iguales pasaban
        la validación de forma —no hay destructivos y entran en el
        presupuesto— así que el cálculo se preparaba DOS veces.

        Idénticos significa misma acción **y** mismos parámetros. Dos
        `preparar_calculo` con fórmulas distintas son dos cálculos que el
        usuario pidió; dos con la misma fórmula son uno solo dicho dos
        veces. Comparar solo la acción rompería los planes legítimos.

        Solo CONSECUTIVOS: `listar_archivos(/a)`, `crear_directorio(/b)`,
        `listar_archivos(/a)` es una repetición pedida a propósito —mirar
        antes y después— y ahí los pasos iguales no se tocan.

        **Y solo si el paso dice sobre qué material trabaja.** Dos pasos
        pueden verse idénticos porque el modelo perdió justo el dato que
        los distingue: "relajá ZrO2 en bcc y en fcc" puede salir como dos
        `preparar_calculo(tipo_calculo=relajacion)` sin `formula`, que son
        DOS cálculos, no uno tartamudeado. Fusionarlos se comería un
        cálculo y además borraría la señal de la que vive la recuperación
        —`BecarioService._needs_decomposition` busca exactamente un plan
        multi-paso con un cálculo sin material para volver a pedirlo
        descompuesto—. Un valor ausente no puede colapsar dos estados
        distintos: ante la duda, no se toca.

        Se colapsa en vez de rechazar porque `parse_llm_output` es
        fail-closed: elevar acá convertiría un pedido que el modelo
        entendió bien en un "no te entendí". El usuario dijo una cosa una
        vez, y el plan que ve confirma exactamente eso.
        """
        if not steps:
            return steps
        collapsed = [steps[0]]
        for step in steps[1:]:
            if step != collapsed[-1] or _sin_material(step):
                collapsed.append(step)
        return collapsed

    @field_validator("steps")
    @classmethod
    def _v_destructive_last(cls, steps: list[PlanStep]) -> list[PlanStep]:
        idx = [i for i, s in enumerate(steps) if s.action in Intent.destructive()]
        if len(idx) > 1:
            raise ValueError("un plan admite a lo sumo un paso destructivo")
        if idx and idx[0] != len(steps) - 1:
            raise ValueError("el paso destructivo debe ser el último del plan")
        return steps

    @property
    def is_single(self) -> bool:
        return len(self.steps) == 1

    @property
    def single_step(self) -> PlanStep:
        return self.steps[0]


@dataclass(frozen=True)
class CommandResult:
    """Resultado de ejecutar algo en el cluster.

    `job_id` solo lo completa `submit_job` cuando pudo parsear el ID que
    devuelve `sbatch` — es lo que permite después rastrear el trabajo.
    """

    ok: bool
    stdout: str = ""
    stderr: str = ""
    job_id: Optional[str] = None

    @property
    def message(self) -> str:
        return self.stdout.strip() or self.stderr.strip() or "(sin salida)"


# ---------------------------------------------------------------------------
# Seguimiento de trabajos (cierre del loop)
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Estado normalizado de un trabajo, independiente de cómo Slurm lo
    reporte exactamente (sacct usa strings como 'CANCELLED by 12345')."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def terminal(cls) -> frozenset["JobStatus"]:
        """Estados en los que el trabajo ya no va a cambiar: acá es cuando
        corresponde notificar y dejar de rastrearlo."""
        return frozenset({cls.COMPLETED, cls.FAILED, cls.CANCELLED, cls.TIMEOUT})

    @property
    def label_es(self) -> str:
        """Nombre del estado para mostrarle al usuario (en español)."""
        return {
            JobStatus.PENDING: "en cola",
            JobStatus.RUNNING: "corriendo",
            JobStatus.COMPLETED: "completado",
            JobStatus.FAILED: "falló",
            JobStatus.CANCELLED: "cancelado",
            JobStatus.TIMEOUT: "agotó el tiempo límite",
            JobStatus.UNKNOWN: "desconocido",
        }[self]

    @classmethod
    def from_slurm(cls, raw: Optional[str]) -> "JobStatus":
        """Traduce el campo State de sacct a nuestro estado normalizado.
        Vive en el dominio (no en el gateway SSH) porque decidir qué cuenta
        como terminal es política de negocio, no un detalle de Slurm."""
        text = (raw or "").strip().upper()
        if text.startswith("CANCELLED"):  # sacct: "CANCELLED by 12345"
            return cls.CANCELLED
        mapping = {
            "COMPLETED": cls.COMPLETED,
            "FAILED": cls.FAILED,
            "TIMEOUT": cls.TIMEOUT,
            "OUT_OF_MEMORY": cls.FAILED,
            "NODE_FAIL": cls.FAILED,
            "BOOT_FAIL": cls.FAILED,
            "DEADLINE": cls.FAILED,
            "PREEMPTED": cls.FAILED,
            "REVOKED": cls.FAILED,
            "PENDING": cls.PENDING,
            "RUNNING": cls.RUNNING,
            "COMPLETING": cls.RUNNING,
            "CONFIGURING": cls.PENDING,
            "SUSPENDED": cls.RUNNING,
            "REQUEUED": cls.PENDING,
            "RESIZING": cls.RUNNING,
        }
        return mapping.get(text, cls.UNKNOWN)


@dataclass
class TrackedJob:
    """Un trabajo enviado por BECARIO que el monitor sigue hasta que
    termina. `notified=False` es la única condición para seguir
    consultándolo — una vez notificado, se deja de rastrear."""

    job_id: str
    owner_id: int
    chat_id: int
    ssh_user: str
    job_name: str
    script_path: str = ""  # ruta remota del script: su directorio es la corrida
    workflow: str = ""  # "" = job suelto; "encut_scan" = cosecha al terminar
    status: JobStatus = JobStatus.PENDING
    notified: bool = False
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Confirmaciones pendientes
# ---------------------------------------------------------------------------


@dataclass
class PendingAction:
    """Acción destructiva a la espera de confirmación del usuario.

    `requester_id` (telegram_user_id) es quien la pidió: solo esa persona
    puede confirmarla o rechazarla, aunque el chat fuera compartido.

    `request_intent`/`request_params` conservan el pedido ORIGINAL (el que
    interpretó el LLM), para poder "modificar el plan": se mezclan con lo
    nuevo que diga el usuario y se rearma la confirmación. Si
    `request_intent` es None, la acción no admite modificación (p. ej.
    cancelar un job).
    """

    chat_id: int
    requester_id: int
    intent: Intent
    description: str
    payload: dict
    request_intent: Optional[Intent] = None
    request_params: dict = field(default_factory=dict)
    token: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)

    def expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


@dataclass
class PendingPlan:
    """Plan a la espera de confirmación: un solo token/TTL para todo el
    plan, con sus pasos ya materializados como `PendingAction`.

    Generaliza `PendingAction` a N pasos sin duplicar su semántica de
    ownership/expiración: un plan de un solo paso destructivo es
    indistinguible, en estos aspectos, de la `PendingAction` de hoy.
    """

    chat_id: int
    requester_id: int
    steps: list[PendingAction]
    token: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)
    # Id de la decisión del router que originó este plan (ver
    # `RouterDecisionLog`): permite etiquetar el ruteo con el desenlace
    # humano (confirmar/cancelar). None si el log está apagado.
    decision_id: Optional[int] = None
    # Batch (ADR-0007): si es True, confirmar ejecuta TODOS los pasos en
    # orden (nada se materializó antes), en vez de solo la cola destructiva
    # final. `payload` de cada paso lleva sus parámetros crudos para ejecutar
    # recién al confirmar.
    execute_all: bool = False

    def expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.created_at) > ttl_seconds

    @property
    def allow_modify(self) -> bool:
        """El plan es editable si al menos un paso conserva el pedido
        original (`request_intent`), igual que hoy con `PendingAction`."""
        return any(s.request_intent is not None for s in self.steps)
