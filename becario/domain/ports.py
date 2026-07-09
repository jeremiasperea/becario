"""Puertos (interfaces) del dominio.

Principio de Inversión de Dependencias: la capa de aplicación depende de
estas abstracciones; paramiko, Ollama, SQLite y Telegram son detalles que
las implementan en `infrastructure/`.
"""
from __future__ import annotations

from typing import Optional, Protocol

from .models import (
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
)


class IntentRouter(Protocol):
    """Traduce texto libre del usuario a una intención estructurada."""

    def route(self, user_text: str) -> RoutedRequest: ...


class StructureBuilder(Protocol):
    """Construye estructuras atómicas y escribe archivos de entrada
    (POSCAR/CIF/XYZ) localmente. Implementado con ASE."""

    def build(self, req: StructureRequest) -> StructureResult: ...


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

    def upload_file(self, local_path: str, remote_path: str) -> CommandResult: ...


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
