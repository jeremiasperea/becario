"""Modelos de dominio de B.E.C.A.R.I.O.

Toda la validación/sanitización de parámetros vive acá, en el centro de la
arquitectura. Las capas externas (Telegram, SSH, LLM) nunca construyen
comandos a partir de strings sin pasar por estos modelos.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

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


class CalcKind(str, Enum):
    """Tipos de cálculo VASP que BECARIO sabe preparar."""

    RELAX = "relajacion"  # ISIF=3: relaja también los parámetros de red
    STATIC = "estatico"
    ENCUT_SCAN = "convergencia_encut"


class OutputFormat(str, Enum):
    VASP = "vasp"  # POSCAR
    CIF = "cif"
    XYZ = "xyz"


class StructureRequest(BaseModel):
    """Pedido validado de construcción de una estructura atómica."""

    formula: str
    kind: StructureKind = StructureKind.BULK
    crystal: Optional[str] = None  # p. ej. diamond, fcc, rocksalt, zincblende
    lattice_a: Optional[float] = Field(default=None, gt=0.5, lt=50.0)  # Å
    supercell: tuple[int, int, int] = (1, 1, 1)
    vacuum: Optional[float] = Field(default=None, ge=0.0, le=60.0)  # Å
    output_format: OutputFormat = OutputFormat.VASP
    remote_dest_dir: Optional[str] = None  # si se define, se sube por SFTP

    @field_validator("formula")
    @classmethod
    def _v_formula(cls, v: str) -> str:
        v = v.strip()
        if not _FORMULA_RE.match(v):
            raise ValueError(f"fórmula inválida: {v!r}")
        return v

    @field_validator("crystal")
    @classmethod
    def _v_crystal(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if not re.fullmatch(r"[a-z]{2,20}", v):
            raise ValueError(f"red cristalina inválida: {v!r}")
        return v

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
        if not _SCRIPT_PATH_RE.match(v) or ".." in v:
            raise ValueError(f"directorio remoto inválido: {v!r}")
        return v


@dataclass(frozen=True)
class StructureResult:
    """Estructura construida: archivo local + resumen físico."""

    local_path: str
    filename: str
    chemical_formula: str
    n_atoms: int
    cell_summary: str
    uploaded_to: Optional[str] = None

    def describe(self) -> str:
        lines = [
            f"fórmula: {self.chemical_formula}",
            f"átomos: {self.n_atoms}",
            f"celda: {self.cell_summary}",
            f"archivo: {self.filename}",
        ]
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
    calc_kind: CalcKind = CalcKind.STATIC
    encut: int = Field(default=520, ge=100, le=1500)  # eV
    kpoints: Optional[tuple[int, int, int]] = None  # None => grilla automática
    encut_values: Optional[list[int]] = None  # solo para ENCUT_SCAN
    partition: str = Field(default="default")
    nodes: int = Field(default=1, ge=1, le=64)
    time_limit: str = Field(default="01:00:00")

    @field_validator("formula")
    @classmethod
    def _v_formula(cls, v: str) -> str:
        v = v.strip()
        if not _FORMULA_RE.match(v):
            raise ValueError(f"fórmula inválida: {v!r}")
        return v

    @field_validator("crystal")
    @classmethod
    def _v_crystal(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if not re.fullmatch(r"[a-z]{2,20}", v):
            raise ValueError(f"red cristalina inválida: {v!r}")
        return v

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
