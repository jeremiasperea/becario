"""Persistencia: historial SQLite (parametrizado) y confirmaciones en memoria."""
from __future__ import annotations

import sqlite3
import threading
from typing import Optional

from ..domain.models import HistoryFilter, JobStatus, PendingAction, TrackedJob


class SQLiteHistoryRepository:
    """Historial de cálculos del grupo. Todas las queries usan placeholders
    `?`: la inyección SQL queda estructuralmente imposible.

    `owner_id` identifica al telegram_user_id dueño del registro; `search`
    siempre lo usa para acotar a "lo mío" salvo que `flt.owner_id` sea None
    (uso administrativo fuera del bot, no alcanzable desde Telegram)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

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


class InMemoryConfirmationStore:
    """Acciones pendientes con TTL. Thread-safe (PTB usa asyncio pero los
    handlers pueden intercalarse; el lock es barato y elimina la duda)."""

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def put(self, action: PendingAction) -> str:
        with self._lock:
            self._items[action.token] = action
        return action.token

    def peek(self, token: str) -> Optional[PendingAction]:
        with self._lock:
            action = self._items.get(token)
        if action is None or action.expired(self._ttl):
            return None
        return action

    def pop(self, token: str) -> Optional[PendingAction]:
        with self._lock:
            action = self._items.pop(token, None)
        if action is None or action.expired(self._ttl):
            return None
        return action

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
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    notified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (job_id, owner_id)
                )
                """
            )

    def track(self, job: TrackedJob) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trabajos_monitoreados
                    (job_id, owner_id, chat_id, ssh_user, job_name, status, notified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id, job.owner_id, job.chat_id, job.ssh_user,
                    job.job_name, job.status.value, int(job.notified),
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
