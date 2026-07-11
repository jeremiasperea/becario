"""Puertos (interfaces) del dominio.

Principio de Inversión de Dependencias: la capa de aplicación depende de
estas abstracciones; paramiko, Ollama, SQLite y Telegram son detalles que
las implementan en `infrastructure/`.
"""
from __future__ import annotations

from typing import Optional, Protocol

from .models import (
    CalcDirResult,
    ClusterIdentity,
    CommandResult,
    HistoryFilter,
    JobId,
    JobStatus,
    PendingAction,
    RoutedRequest,
    SlurmJobRequest,
    StructureRequest,
    StructureResult,
    TrackedJob,
    VaspCalcRequest,
)


class IntentRouter(Protocol):
    """Traduce texto libre del usuario a una intención estructurada."""

    def route(self, user_text: str) -> RoutedRequest: ...

    def extract_params(self, user_text: str) -> dict:
        """Solo parámetros, sin decidir acción: para mensajes que describen
        un CAMBIO sobre un pedido ya armado («usá 2 nodos», «subí el ENCUT
        a 600»). Devuelve {} si no encontró nada."""
        ...


class StructureBuilder(Protocol):
    """Construye estructuras atómicas y escribe archivos de entrada
    (POSCAR/CIF/XYZ) localmente. Implementado con ASE."""

    def build(self, req: StructureRequest) -> StructureResult: ...


class CalcInputGenerator(Protocol):
    """Genera localmente el directorio completo de una corrida VASP
    (POSCAR/INCAR/KPOINTS/script), listo para subir al cluster."""

    def generate(self, req: VaspCalcRequest) -> CalcDirResult: ...


class ClusterGateway(Protocol):
    """Operaciones sobre el cluster HPC, atadas a la cuenta SSH de UN usuario.
    Cada método construye su propio comando a partir de parámetros ya
    validados — nunca acepta strings de shell arbitrarios."""

    def submit_job(self, req: SlurmJobRequest) -> CommandResult: ...

    def cancel_job(self, job_id: JobId) -> CommandResult: ...

    def job_status(self, job_id: Optional[JobId]) -> CommandResult: ...

    def job_state(self, job_id: JobId) -> Optional[str]:
        """Estado crudo de Slurm (una palabra, vía `sacct --parsable2`),
        pensado para que el monitor lo interprete — no para mostrar al
        usuario. None si no se pudo determinar."""
        ...

    def make_directory(self, path: str) -> CommandResult:
        """Crea un directorio remoto (mkdir -p: idempotente, con padres).
        La ruta ya viene validada por el dominio (`RemoteDirRequest`)."""
        ...

    def list_directory(self, path: str) -> CommandResult:
        """Listado legible de un directorio remoto (solo lectura).
        La ruta ya viene validada por el dominio (`ListFilesRequest`)."""
        ...

    def upload_file(self, local_path: str, remote_path: str) -> CommandResult: ...

    def upload_dir(self, local_dir: str, remote_dir: str) -> CommandResult:
        """Sube un directorio completo (recursivo) por SFTP."""
        ...

    def home_dir(self) -> Optional[str]:
        """Home remoto de la cuenta, para resolver rutas absolutas."""
        ...

    def file_exists(self, remote_path: str) -> bool: ...

    def list_dir(self, remote_dir: str) -> Optional[list[str]]:
        """Nombres dentro de un directorio remoto; None si no se pudo leer."""
        ...

    def read_file(self, remote_path: str) -> Optional[str]:
        """Contenido de un archivo remoto chico (OSZICAR, POSCAR…);
        None si no existe o no se pudo leer."""
        ...

    def concat_files(self, sources: list[str], dest: str) -> CommandResult:
        """`cat` remoto de varios archivos en uno (armar POTCAR desde la
        biblioteca del cluster). Rutas ya validadas por el dominio."""
        ...


class ClusterGatewayFactory(Protocol):
    """Resuelve el gateway (conexión SSH) correspondiente a cada identidad.

    Nunca hay una conexión "global": siempre se pide un gateway atado a la
    cuenta de una persona concreta, para que el aislamiento entre usuarios
    quede garantizado por el propio sistema del cluster."""

    def for_identity(self, identity: ClusterIdentity) -> ClusterGateway: ...


class UserRegistry(Protocol):
    """Roster del grupo: quién es quién en el cluster.

    No expone altas/bajas — eso lo administra quien mantiene el grupo con
    `scripts/manage_users.py`, fuera de Telegram (no hay rol admin en el bot).
    """

    def get_identity(self, telegram_user_id: int) -> Optional[ClusterIdentity]: ...


class HistoryRepository(Protocol):
    """Historial de cálculos del grupo (queries parametrizadas)."""

    def add(self, owner_id: int, job_id: str, nombre_trabajo: str, estado: str) -> None: ...

    def search(self, flt: HistoryFilter) -> list[dict]: ...


class JobTracker(Protocol):
    """Trabajos enviados por BECARIO que el monitor sigue hasta que
    terminan (ver `application/job_monitor.py`)."""

    def track(self, job: TrackedJob) -> None: ...

    def active_jobs(self) -> list[TrackedJob]:
        """Trabajos todavía no notificados — lo único que el monitor
        necesita volver a consultar."""
        ...

    def update_status(self, job_id: str, owner_id: int, status: JobStatus) -> None: ...

    def mark_notified(self, job_id: str, owner_id: int) -> None: ...


class CalcRunRepository(Protocol):
    """Corridas VASP ya enviadas por cada persona, con una huella de sus
    parámetros: permite avisar cuando algo idéntico o muy similar ya se
    corrió antes de volver a gastarle horas al cluster."""

    def add(
        self, owner_id: int, job_id: str, job_name: str,
        fingerprint: str, run_dir: str,
    ) -> None: ...

    def find_by_name(
        self, owner_id: int, job_name: str, limit: int = 3
    ) -> list[dict]:
        """Corridas previas del MISMO dueño con el mismo job_name
        (material + tipo de cálculo), más recientes primero."""
        ...

    def find_recent(
        self, owner_id: int, job_name_prefix: str = "", limit: int = 5
    ) -> list[dict]:
        """Corridas del dueño cuyo job_name empieza con el prefijo (p. ej.
        'Zr_' = cualquier cálculo de Zr), más recientes primero."""
        ...


class ConfirmationStore(Protocol):
    """Guarda acciones destructivas pendientes de confirmación."""

    def put(self, action: PendingAction) -> str: ...

    def peek(self, token: str) -> Optional[PendingAction]:
        """Consulta sin consumir — para validar quién puede confirmar antes
        de descartar la acción pendiente."""
        ...

    def pop(self, token: str) -> Optional[PendingAction]: ...

    def purge_expired(self) -> int: ...


class Transcriber(Protocol):
    """Transcripción de audio a texto (Whisper u otro backend)."""

    def transcribe(self, audio_bytes: bytes) -> str: ...
