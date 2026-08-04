"""Puertos (interfaces) del dominio.

Principio de Inversión de Dependencias: la capa de aplicación depende de
estas abstracciones; paramiko, Ollama, SQLite y Telegram son detalles que
las implementan en `infrastructure/`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from .models import (
    CalcDirResult,
    ClusterIdentity,
    CommandResult,
    HistoryFilter,
    JobId,
    JobStatus,
    PendingPlan,
    Plan,
    SlurmJobRequest,
    StructureQuery,
    StructureRequest,
    StructureResolution,
    StructureResult,
    TrackedJob,
    VaspCalcRequest,
)

if TYPE_CHECKING:  # solo para anotar la firma; el dominio no importa ASE
    from ase import Atoms


class IntentRouter(Protocol):
    """Traduce texto libre del usuario a un plan estructurado (1 a 5
    pasos; un pedido de una sola acción es un plan de un solo paso)."""

    def route(self, user_text: str) -> Plan: ...

    def extract_params(self, user_text: str) -> dict:
        """Solo parámetros, sin decidir acción: para mensajes que describen
        un CAMBIO sobre un pedido ya armado («usá 2 nodos», «subí el ENCUT
        a 600»). Devuelve {} si no encontró nada."""
        ...

    def extract_edit(self, plan_context: str, user_text: str) -> tuple[Optional[int], dict]:
        """Ídem `extract_params`, pero para un cambio sobre un plan de
        VARIOS pasos: además del delta de parámetros, intenta identificar
        a qué paso (1-based, `target_index`) se refiere el mensaje —
        semántico o explícito («paso N»). `target_index=None` si no hay
        confianza sobre a cuál paso se refiere; nunca se adivina."""
        ...


class StructureBuilder(Protocol):
    """Construye estructuras atómicas y escribe archivos de entrada
    (POSCAR/CIF/XYZ) localmente. Implementado con ASE."""

    def build(self, req: StructureRequest) -> StructureResult: ...


class CalcInputGenerator(Protocol):
    """Genera localmente el directorio completo de una corrida VASP
    (POSCAR/INCAR/KPOINTS/script), listo para subir al cluster."""

    def generate(
        self, req: VaspCalcRequest, atoms: "Atoms | None" = None
    ) -> CalcDirResult: ...


class StructureProvider(Protocol):
    """Resuelve una estructura cristalina desde una fuente externa
    (Materials Project) para compuestos que ASE no sabe armar.

    La capa de aplicación arma el `StructureQuery` y decide cuándo usar este
    puerto (compuestos) vs el `StructureBuilder`/ASE (elementos). El proveedor
    devuelve `ase.Atoms` + metadatos, sin filtrar tipos de pymatgen al dominio,
    y lanza `StructureResolutionError` ante fallos de red/API/sin-resultado."""

    def resolve(self, query: StructureQuery) -> StructureResolution: ...


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

    def file_exists(self, remote_path: str) -> Optional[bool]:
        """¿Existe el archivo remoto? `None` si NO SE PUDO AVERIGUAR.

        Los tres desenlaces son distintos y quien llama necesita separarlos:
        'no está' se arregla poniendo el archivo, 'no pude preguntar' se
        arregla levantando el cluster. Colapsarlos en `False` mandaba al
        usuario a revisar el lugar equivocado.

        Misma convención que `list_dir`: `None` es 'no se pudo'."""
        ...

    def list_dir(self, remote_dir: str) -> Optional[list[str]]:
        """Nombres dentro de un directorio remoto; None si no se pudo leer."""
        ...

    def read_file(
        self, remote_path: str, max_bytes: Optional[int] = None
    ) -> Optional[str]:
        """Contenido de un archivo remoto chico (OSZICAR, POSCAR…);
        None si no existe o no se pudo leer. `max_bytes` acota cuánto se
        lee (y por ende cuánta memoria se usa): imprescindible cuando el
        nombre lo elige el usuario, para no descargar entero un archivo
        gigante (WAVECAR/CHGCAR pueden pesar GB). None = sin límite."""
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


class ChatLogRepository(Protocol):
    """Bitácora de la conversación de cada chat (mensajes del usuario y
    respuestas del bot). Se guarda todo, sin límite de retención: sirve
    como registro auditable de lo que se pidió y lo que se respondió."""

    def add(self, chat_id: int, role: str, text: str) -> None:
        """Registra un mensaje. `role` es 'user' o 'bot'."""
        ...

    def recent(self, chat_id: int, limit: int = 50) -> list[dict]:
        """Últimos mensajes del chat, del más viejo al más nuevo (orden
        de lectura natural del historial)."""
        ...


class RouterDecisionLog(Protocol):
    """Registro de cada decisión del enrutador y de su desenlace humano.

    Es la materia prima del set de evaluación del router: cada mensaje
    real queda asociado al plan que el LLM produjo, y el mecanismo de
    confirmación de ADR-0003 etiqueta gratis — confirmar un plan avala el
    ruteo; cancelarlo lo pone en duda. Las decisiones sin desenlace
    explícito quedan en 'routed' para revisión manual."""

    def add(
        self, chat_id: int, user_id: int, text: str,
        steps_json: str, latency_seconds: float,
    ) -> int:
        """Registra una decisión recién ruteada y devuelve su id.
        `steps_json` es la lista de pasos serializada (action + parámetros).
        Qué modelo la produjo es detalle del adaptador (lo sabe la
        composition root, no el caso de uso)."""
        ...

    def set_outcome(self, decision_id: int, outcome: str) -> None:
        """Marca el desenlace: 'confirmed', 'cancelled' o 'error'."""
        ...


class ConfirmationStore(Protocol):
    """Guarda planes (uno o más pasos) pendientes de confirmación."""

    def put(self, plan: PendingPlan) -> str: ...

    def peek(self, token: str) -> Optional[PendingPlan]:
        """Consulta sin consumir — para validar quién puede confirmar antes
        de descartar el plan pendiente."""
        ...

    def pop(self, token: str) -> Optional[PendingPlan]: ...

    def status(self, token: str) -> str:
        """Por qué `peek` devolvió `None`: `"vigente"`, `"vencido"`,
        `"consumido"` o `"desconocido"` (nunca visto por este proceso).

        `peek` colapsa dos situaciones distintas en `None`, y quien las
        muestra necesita separarlas: un plan que venció por TTL se rehace,
        uno que ya se consumió significa que la acción YA ocurrió — y decirle
        "expiró" a alguien que acaba de apretar el botón dos veces lo manda a
        empezar de nuevo algo que en realidad salió bien."""
        ...

    def purge_expired(self) -> int: ...


class Transcriber(Protocol):
    """Transcripción de audio a texto (Whisper u otro backend)."""

    def transcribe(self, audio_bytes: bytes) -> str: ...
