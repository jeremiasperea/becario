"""Persistencia: historial SQLite (parametrizado) y confirmaciones en memoria."""
from __future__ import annotations

import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from ..domain.models import HistoryFilter, JobStatus, PendingPlan, TrackedJob

# Cuánto espera un escritor a que se libere la base antes de fallar con
# «database is locked». Con WAL las esperas reales son de milisegundos;
# el margen amplio cubre picos de escritura (p. ej. bitácora de chat).
_BUSY_TIMEOUT_MS = 5000


def _connect(db_path: str) -> sqlite3.Connection:
    """Conexión SQLite endurecida para acceso concurrente.

    Todos los repositorios comparten el mismo archivo, así que la
    configuración vive acá y no en cada clase:

    - `journal_mode=WAL`: lectores y escritor no se bloquean entre sí.
      Es persistente en el archivo, pero se aplica en cada conexión para
      cubrir bases creadas antes de este cambio.
    - `busy_timeout`: ante contención, esperar en vez de fallar al toque.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn


class SQLiteHistoryRepository:
    """Historial de cálculos del grupo. Todas las queries usan placeholders
    `?`: la inyección SQL queda estructuralmente imposible.

    `owner_id` identifica al telegram_user_id dueño del registro; `search`
    siempre lo usa para acotar a "lo mío" salvo que `flt.owner_id` sea None
    (uso administrativo fuera del bot, no alcanzable desde Telegram)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return _connect(self._db_path)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historial_calculos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER,
                    job_id TEXT,
                    nombre_trabajo TEXT,
                    estado TEXT,
                    fecha TEXT DEFAULT (datetime('now'))
                )
                """
            )

    def add(self, owner_id: int, job_id: str, nombre_trabajo: str, estado: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO historial_calculos (owner_id, job_id, nombre_trabajo, estado) "
                "VALUES (?, ?, ?, ?)",
                (owner_id, job_id, nombre_trabajo, estado),
            )

    def search(self, flt: HistoryFilter) -> list[dict]:
        sql = "SELECT * FROM historial_calculos"
        clauses: list[str] = []
        args: list = []
        if flt.owner_id is not None:
            clauses.append("owner_id = ?")
            args.append(flt.owner_id)
        if flt.job_id:
            clauses.append("job_id = ?")
            args.append(flt.job_id)
        elif flt.name_contains:
            clauses.append("nombre_trabajo LIKE ?")
            args.append(f"%{flt.name_contains}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY fecha DESC LIMIT ?"
        args.append(flt.limit)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(row) for row in rows]


class SQLiteCalcRunRepository:
    """Corridas VASP enviadas (implementa CalcRunRepository). La huella
    (`fingerprint`) es el JSON canónico de los parámetros del cálculo:
    huellas iguales = pedido idéntico; mismo job_name = muy similar."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return _connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corridas_vasp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER NOT NULL,
                    job_id TEXT,
                    job_name TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    fecha TEXT DEFAULT (datetime('now'))
                )
                """
            )

    def add(
        self, owner_id: int, job_id: str, job_name: str,
        fingerprint: str, run_dir: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO corridas_vasp (owner_id, job_id, job_name, fingerprint, run_dir) "
                "VALUES (?, ?, ?, ?, ?)",
                (owner_id, job_id, job_name, fingerprint, run_dir),
            )

    def find_by_name(
        self, owner_id: int, job_name: str, limit: int = 3
    ) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM corridas_vasp WHERE owner_id = ? AND job_name = ? "
                "ORDER BY fecha DESC, id DESC LIMIT ?",
                (owner_id, job_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_recent(
        self, owner_id: int, job_name_prefix: str = "", limit: int = 5
    ) -> list[dict]:
        # El prefijo viene de una fórmula ya validada ([A-Za-z0-9]): no
        # puede colar comodines de LIKE.
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM corridas_vasp WHERE owner_id = ? AND job_name LIKE ? "
                "ORDER BY fecha DESC, id DESC LIMIT ?",
                (owner_id, f"{job_name_prefix}%", limit),
            ).fetchall()
        return [dict(row) for row in rows]


class SQLiteChatLogRepository:
    """Bitácora de conversación por chat (implementa ChatLogRepository).
    Usa el mismo archivo que el resto de la persistencia, en una tabla
    propia. Se guarda todo, sin retención: es un registro auditable."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return _connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'bot')),
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_fecha "
                "ON chat_messages (chat_id, created_at)"
            )

    def add(self, chat_id: int, role: str, text: str) -> None:
        # ISO-8601 en UTC generado acá (y no con datetime('now') de SQLite)
        # para que el formato quede idéntico al del resto del código Python.
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (chat_id, role, text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (chat_id, role, text, created_at),
            )

    def recent(self, chat_id: int, limit: int = 50) -> list[dict]:
        # Se piden los N más nuevos y se invierte el resultado: el
        # historial se lee del más viejo al más nuevo.
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE chat_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]


class SQLiteRouterDecisionLog:
    """Registro de decisiones del router (implementa RouterDecisionLog).
    Cada mensaje ruteado guarda el plan que el LLM produjo y, cuando el
    flujo de confirmación lo resuelve, su desenlace. Es la materia prima
    del set de evaluación del router: `scripts/export_router_dataset.py`
    lo convierte en fixtures/JSONL."""

    def __init__(self, db_path: str, model: str) -> None:
        self._db_path = db_path
        self._model = model
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return _connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisiones_router (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    latency_seconds REAL NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'routed'
                        CHECK (outcome IN ('routed', 'confirmed', 'cancelled', 'error')),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decisiones_router_outcome "
                "ON decisiones_router (outcome, created_at)"
            )

    def add(
        self, chat_id: int, user_id: int, text: str,
        steps_json: str, latency_seconds: float,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO decisiones_router "
                "(chat_id, user_id, text, steps_json, model, latency_seconds, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_id, user_id, text, steps_json, self._model,
                 latency_seconds, created_at),
            )
            return int(cur.lastrowid)

    def set_outcome(self, decision_id: int, outcome: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE decisiones_router SET outcome = ? WHERE id = ?",
                (outcome, decision_id),
            )

    def rows(self, outcome: Optional[str] = None) -> list[dict]:
        """Decisiones registradas, opcionalmente filtradas por desenlace.
        No es parte del puerto (el caso de uso solo escribe): lo usan el
        script de exportación del dataset y los tests."""
        sql = "SELECT * FROM decisiones_router"
        args: list = []
        if outcome is not None:
            sql += " WHERE outcome = ?"
            args.append(outcome)
        sql += " ORDER BY id"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]


class InMemoryConfirmationStore:
    """Planes pendientes con TTL. Thread-safe (PTB usa asyncio pero los
    handlers pueden intercalarse; el lock es barato y elimina la duda).

    Duck-typed: solo usa `.token`/`.expired()` de lo que guarda, así que
    aceptar `PendingPlan` (en vez del `PendingAction` de un solo paso que
    guardaba antes) no cambió el comportamiento, solo el tipo declarado."""

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, PendingPlan] = {}
        # Ids de tokens ya consumidos, acotado: alcanza para reconocer un
        # doble toque (segundos) sin crecer sin límite en un proceso largo.
        self._consumed: deque[str] = deque(maxlen=256)
        self._lock = threading.Lock()

    def put(self, plan: PendingPlan) -> str:
        with self._lock:
            self._items[plan.token] = plan
        return plan.token

    def peek(self, token: str) -> Optional[PendingPlan]:
        with self._lock:
            plan = self._items.get(token)
        if plan is None or plan.expired(self._ttl):
            return None
        return plan

    def status(self, token: str) -> str:
        """Ver el puerto. Cuatro respuestas, no tres: hay tokens que este
        proceso no vio nunca —el store es en memoria, así que un reinicio
        los borra— y de esos NO se puede afirmar que se usaron. Decir "ya
        se hizo" sobre algo que no nos consta es la clase de mentira
        creíble que este código viene evitando."""
        with self._lock:
            plan = self._items.get(token)
            consumido = token in self._consumed
        if plan is not None:
            return "vencido" if plan.expired(self._ttl) else "vigente"
        return "consumido" if consumido else "desconocido"

    def pop(self, token: str) -> Optional[PendingPlan]:
        with self._lock:
            plan = self._items.pop(token, None)
            if plan is not None:
                # Lápida acotada: solo para poder distinguir después "ya se
                # usó" de "no lo conozco". No guarda el plan, solo el id.
                self._consumed.append(token)
        if plan is None or plan.expired(self._ttl):
            return None
        return plan

    def purge_expired(self) -> int:
        with self._lock:
            stale = [t for t, a in self._items.items() if a.expired(self._ttl)]
            for token in stale:
                del self._items[token]
        return len(stale)


class SQLiteJobTracker:
    """Trabajos enviados por BECARIO en seguimiento hasta que terminan
    (implementa JobTracker). Usa el mismo archivo que el historial, en
    una tabla separada — son conceptos distintos: esta es una cola de
    trabajo interna del monitor, el historial es de cara al usuario."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return _connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trabajos_monitoreados (
                    job_id TEXT NOT NULL,
                    owner_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    ssh_user TEXT NOT NULL,
                    job_name TEXT NOT NULL,
                    script_path TEXT NOT NULL DEFAULT '',
                    workflow TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    notified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (job_id, owner_id)
                )
                """
            )
            # Migración suave: bases creadas antes de estas columnas.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(trabajos_monitoreados)")}
            for column in ("script_path", "workflow"):
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE trabajos_monitoreados "
                        f"ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                    )

    def track(self, job: TrackedJob) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trabajos_monitoreados
                    (job_id, owner_id, chat_id, ssh_user, job_name, script_path,
                     workflow, status, notified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id, job.owner_id, job.chat_id, job.ssh_user,
                    job.job_name, job.script_path, job.workflow,
                    job.status.value, int(job.notified),
                ),
            )

    def active_jobs(self) -> list[TrackedJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trabajos_monitoreados WHERE notified = 0"
            ).fetchall()
        return [
            TrackedJob(
                job_id=r["job_id"], owner_id=r["owner_id"], chat_id=r["chat_id"],
                ssh_user=r["ssh_user"], job_name=r["job_name"],
                script_path=r["script_path"], workflow=r["workflow"],
                status=JobStatus(r["status"]), notified=bool(r["notified"]),
            )
            for r in rows
        ]

    def update_status(self, job_id: str, owner_id: int, status: JobStatus) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trabajos_monitoreados SET status = ? WHERE job_id = ? AND owner_id = ?",
                (status.value, job_id, owner_id),
            )

    def mark_notified(self, job_id: str, owner_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trabajos_monitoreados SET notified = 1 WHERE job_id = ? AND owner_id = ?",
                (job_id, owner_id),
            )
