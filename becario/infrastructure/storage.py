"""Persistencia: historial SQLite (parametrizado) y confirmaciones."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional

from ..domain.models import (
    HistoryFilter,
    Intent,
    JobStatus,
    PendingAction,
    PendingEdit,
    PendingPlan,
    TrackedJob,
)

logger = logging.getLogger(__name__)

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

    def find_by_job_id(self, owner_id: int, job_id: str) -> Optional[dict]:
        # `owner_id` va en el WHERE: el número de job lo dice el usuario y
        # es adivinable, así que el aislamiento tiene que estar en la
        # consulta y no en un `if` después.
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM corridas_vasp WHERE owner_id = ? AND job_id = ? "
                "ORDER BY fecha DESC, id DESC LIMIT 1",
                (owner_id, str(job_id)),
            ).fetchone()
        return dict(row) if row is not None else None


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
            # Los fallos van en su PROPIA tabla, no como un `outcome` más.
            # Dos razones. Una: no hubo decisión que registrar — no hay pasos,
            # no hay latencia comparable, y meterlos en `decisiones_router`
            # ensuciaría las estadísticas del router con fallos que no son
            # suyos. Dos: la columna `outcome` tiene un CHECK, y SQLite no
            # sabe alterar un CHECK sin recrear la tabla — pagar una
            # migración para esto sería el precio equivocado.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fallos_router (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reason TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fallos_router_fecha "
                "ON fallos_router (created_at)"
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

    def add_failure(self, reason: str) -> None:
        """Anota que el modelo no llegó a pronunciarse.

        Sin esto, «cuántas veces falló el LLM» no se podía contestar: el
        fallo salía por el log y ahí moría. Y es la pregunta que separa
        «Ollama se cayó una vez» de «hace tres días que no anda».
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fallos_router (reason, model, created_at) VALUES (?, ?, ?)",
                (reason, self._model, datetime.now(timezone.utc).isoformat()),
            )

    def failures(self, since: Optional[str] = None) -> list[dict]:
        """Fallos registrados, opcionalmente desde una fecha ISO. Como
        `rows()`, no es parte del puerto: lo usa el chequeo de salud."""
        sql = "SELECT * FROM fallos_router"
        args: list = []
        if since is not None:
            sql += " WHERE created_at >= ?"
            args.append(since)
        sql += " ORDER BY id"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

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
    guardaba antes) no cambió el comportamiento, solo el tipo declarado.

    **Guarda y devuelve COPIAS, y eso no es paranoia.** `main.py` cablea el
    store de SQLite: esta implementación ya no la usa nadie en producción y
    sobrevive como doble de media suite. El de disco serializa a JSON, así
    que lo que devuelve es siempre un objeto nuevo y mutarlo no toca lo
    guardado. Mientras esta clase devolvía la referencia viva, las dos se
    comportaban distinto ante la misma llamada: un test que mutara un plan
    peekeado pasaba con el doble y fallaba contra el real. Es exactamente
    la forma de los tres bugs que documenta `tests/test_contrato_de_dobles.py`,
    y la única razón de que este fuera el cuarto es que ningún call site de
    producción muta todavía (copian con `dict(...)`).

    El costo es despreciable: un `PendingPlan` es una dataclass de datos
    JSON-ables —por eso el de disco puede serializarla— y se copia una vez
    por toque de botón.
    """

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, PendingPlan] = {}
        # Ids de tokens ya consumidos, acotado: alcanza para reconocer un
        # doble toque (segundos) sin crecer sin límite en un proceso largo.
        self._consumed: deque[str] = deque(maxlen=256)
        self._lock = threading.Lock()

    def put(self, plan: PendingPlan) -> str:
        # Se copia AL GUARDAR además de al leer: quien llama sigue teniendo
        # su `plan` en la mano para armar el mensaje de confirmación, y lo
        # que le haga después no puede alterar lo que se va a ejecutar.
        with self._lock:
            self._items[plan.token] = deepcopy(plan)
        return plan.token

    def peek(self, token: str) -> Optional[PendingPlan]:
        with self._lock:
            plan = self._items.get(token)
        if plan is None or plan.expired(self._ttl):
            return None
        return deepcopy(plan)

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
        """Consume el plan. Sobre uno VENCIDO no hace nada.

        Antes lo sacaba igual y dejaba lápida, así que el siguiente
        `status()` respondía "consumido" y el usuario leía «ya se usó — la
        acción se hizo con el primer toque» sobre algo que nunca corrió:
        la clase de mentira creíble que este store viene evitando (ver
        `status`). Dejarlo donde está hace que siga diciendo "vencido"
        —que manda a rehacer el pedido, lo correcto— hasta que
        `purge_expired` lo levante.
        """
        with self._lock:
            plan = self._items.get(token)
            if plan is None or plan.expired(self._ttl):
                return None
            del self._items[token]
            # Lápida acotada: solo para poder distinguir después "ya se
            # usó" de "no lo conozco". No guarda el plan, solo el id.
            self._consumed.append(token)
        # Se copia igual que en `peek`, aunque acá ya no quede nada adentro:
        # el contrato es "lo que devolvés es tuyo", y que dependa de si el
        # plan seguía o no en el diccionario lo volvería impredecible.
        return deepcopy(plan)

    def purge_expired(self) -> int:
        with self._lock:
            stale = [t for t, a in self._items.items() if a.expired(self._ttl)]
            for token in stale:
                del self._items[token]
        return len(stale)


# ---------------------------------------------------------------------------
# Serialización de planes pendientes
# ---------------------------------------------------------------------------
# `PendingPlan`/`PendingAction` son dataclasses de contenido JSON-able: lo
# único que no lo es son los `Intent`, que viajan por su `.value`. Se
# serializa a mano en vez de con `asdict` para que el formato del archivo
# sea explícito y un cambio en el dominio no lo rompa en silencio.
def _accion_a_dict(a: PendingAction) -> dict:
    return {
        "chat_id": a.chat_id,
        "requester_id": a.requester_id,
        "intent": a.intent.value,
        "description": a.description,
        "payload": a.payload,
        "request_intent": a.request_intent.value if a.request_intent else None,
        "request_params": a.request_params,
        "token": a.token,
        "created_at": a.created_at,
    }


def _accion_desde_dict(d: dict) -> PendingAction:
    crudo = d.get("request_intent")
    return PendingAction(
        chat_id=int(d["chat_id"]),
        requester_id=int(d["requester_id"]),
        intent=Intent(d["intent"]),
        description=d.get("description", ""),
        payload=d.get("payload") or {},
        request_intent=Intent(crudo) if crudo else None,
        request_params=d.get("request_params") or {},
        token=d["token"],
        created_at=float(d["created_at"]),
    )


def _plan_a_json(plan: PendingPlan) -> str:
    return json.dumps({
        "chat_id": plan.chat_id,
        "requester_id": plan.requester_id,
        "steps": [_accion_a_dict(s) for s in plan.steps],
        "token": plan.token,
        "created_at": plan.created_at,
        "decision_id": plan.decision_id,
        "execute_all": plan.execute_all,
    }, ensure_ascii=False)


def _plan_desde_json(raw: str) -> PendingPlan:
    d = json.loads(raw)
    return PendingPlan(
        chat_id=int(d["chat_id"]),
        requester_id=int(d["requester_id"]),
        steps=[_accion_desde_dict(s) for s in d.get("steps", [])],
        token=d["token"],
        created_at=float(d["created_at"]),
        decision_id=d.get("decision_id"),
        execute_all=bool(d.get("execute_all", False)),
    )


class SQLiteConfirmationStore:
    """Planes pendientes de confirmación, en disco (implementa
    `ConfirmationStore`).

    Por qué existe: el store en memoria era lo ÚNICO volátil del sistema
    —historial, trabajos, bitácora y corridas ya viven en SQLite— y era
    justo lo que hacía que una conversación fuera una conversación. Un
    reinicio del bot se llevaba el token en silencio, y el usuario que
    apretaba ✅ cuarenta segundos después de ver la tarjeta recibía
    «expiró o ya fue usada» con un TTL de diez minutos. Pasó de verdad
    (bitácora, mensajes 105-108).

    La lápida de un token consumido se guarda para siempre (sin el plan,
    solo el id y cuándo): es barata y es la que permite decirle a quien
    apretó dos veces «ya se hizo» en vez de «no lo conozco», incluso si el
    proceso se reinició entre los dos toques.
    """

    # Cuánto se conservan las lápidas de tokens consumidos. Un doble toque
    # ocurre en segundos; una semana es holgura para que alguien vuelva al
    # chat el lunes y siga leyendo «ya se hizo».
    _RETENCION_CONSUMIDOS = 7 * 24 * 3600.0

    def __init__(self, db_path: str, ttl_seconds: float = 600.0) -> None:
        self._db_path = db_path
        self._ttl = ttl_seconds
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return _connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS confirmaciones_pendientes (
                    token TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    requester_id INTEGER NOT NULL,
                    plan_json TEXT,
                    created_at REAL NOT NULL,
                    consumed_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_confirmaciones_creado "
                "ON confirmaciones_pendientes (created_at)"
            )

    # ------------------------------------------------------------------
    def put(self, plan: PendingPlan) -> str:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO confirmaciones_pendientes "
                "(token, chat_id, requester_id, plan_json, created_at, consumed_at) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (plan.token, plan.chat_id, plan.requester_id,
                 _plan_a_json(plan), plan.created_at),
            )
        return plan.token

    def _fila(self, conn: sqlite3.Connection, token: str) -> Optional[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM confirmaciones_pendientes WHERE token = ?", (token,)
        ).fetchone()

    def peek(self, token: str) -> Optional[PendingPlan]:
        with self._connect() as conn:
            row = self._fila(conn, token)
        if row is None or row["consumed_at"] is not None or not row["plan_json"]:
            return None
        plan = _plan_desde_json(row["plan_json"])
        return None if plan.expired(self._ttl) else plan

    def status(self, token: str) -> str:
        """Ver el puerto. Con el store en disco «desconocido» pasa a
        significar de verdad «nunca existió», y no «este proceso no lo vio»:
        esa ambigüedad era el síntoma, no la causa."""
        with self._connect() as conn:
            row = self._fila(conn, token)
        if row is None:
            return "desconocido"
        if row["consumed_at"] is not None:
            return "consumido"
        plan_json = row["plan_json"]
        if not plan_json:
            return "vencido"
        return "vencido" if _plan_desde_json(plan_json).expired(self._ttl) else "vigente"

    def pop(self, token: str) -> Optional[PendingPlan]:
        """Consume el token de forma atómica.

        El `UPDATE ... WHERE consumed_at IS NULL` es el que hace que dos
        toques simultáneos no ejecuten dos veces: gana el que cambia la
        fila, el otro ve `rowcount == 0` y se va con las manos vacías. En
        el store en memoria eso lo garantizaba un lock del proceso, que no
        sirve si mañana corren dos.

        Un plan VENCIDO no se marca consumido: nunca se ejecutó, y la
        lápida haría que el próximo `status()` dijera "ya se usó — la
        acción se hizo con el primer toque" sobre algo que no pasó.
        """
        with self._connect() as conn:
            row = self._fila(conn, token)
            if row is None or not row["plan_json"]:
                return None
            plan = _plan_desde_json(row["plan_json"])
            if plan.expired(self._ttl):
                return None
            cur = conn.execute(
                "UPDATE confirmaciones_pendientes SET consumed_at = ?, plan_json = NULL "
                "WHERE token = ? AND consumed_at IS NULL",
                (time.time(), token),
            )
            if cur.rowcount == 0:  # otro lo consumió primero
                return None
        return plan

    def purge_expired(self) -> int:
        """Borra los vencidos sin consumir y, de paso, las lápidas viejas.

        Solo se cuentan los vencidos: es lo que promete el puerto y lo que
        el llamador usa para loguear. Las lápidas se podan en la misma
        pasada porque no hay otro momento natural para hacerlo.
        """
        corte = time.time() - self._ttl
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM confirmaciones_pendientes "
                "WHERE consumed_at IS NULL AND created_at < ?",
                (corte,),
            )
            vencidos = cur.rowcount
            conn.execute(
                "DELETE FROM confirmaciones_pendientes WHERE consumed_at < ?",
                (time.time() - self._RETENCION_CONSUMIDOS,),
            )
        return vencidos


def _edit_a_json(edit: PendingEdit) -> str:
    # Los `Intent` viajan por su `.value`, igual que en los planes; el resto
    # ya es JSON-able. La tupla `(intent, params)` se guarda como lista
    # porque JSON no distingue: se re-arma como tupla al leer.
    return json.dumps({
        "steps": [[i.value, p] for i, p in edit.steps],
        "created_at": edit.created_at,
        "chat_id": edit.chat_id,
        "awaiting_index": edit.awaiting_index,
    }, ensure_ascii=False)


def _edit_desde_json(raw: str) -> PendingEdit:
    d = json.loads(raw)
    return PendingEdit(
        steps=[(Intent(i), p) for i, p in d.get("steps", [])],
        created_at=float(d["created_at"]),
        chat_id=int(d.get("chat_id") or 0),
        awaiting_index=d.get("awaiting_index"),
    )


class SQLitePendingEditStore:
    """Pedidos esperando respuesta, en disco (implementa `PendingEditStore`).

    El caso que lo justifica está en la bitácora: el bot preguntó con qué
    fase de ZrO2 seguir (mensaje 99), el usuario contestó «tetragonal»
    trece minutos después (mensaje 100) —con un TTL de treinta— y recibió
    «No pude interpretar tu pedido». No había vencido: el proceso se
    reinició en el medio y el pendiente vivía en un `dict`.

    Peor que perderlo era no poder saberlo: un pendiente que se fue con el
    proceso no deja rastro, así que el servicio no podía distinguirlo de un
    mensaje que de verdad no se entendió, y respondía lo segundo.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return _connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pedidos_pendientes (
                    user_id INTEGER PRIMARY KEY,
                    edit_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def put(self, user_id: int, edit: PendingEdit) -> None:
        # Uno por usuario: quien contesta una repregunta contesta la última,
        # así que el pedido nuevo pisa al anterior (misma semántica que el
        # `dict` que reemplaza).
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pedidos_pendientes "
                "(user_id, edit_json, created_at) VALUES (?, ?, ?)",
                (user_id, _edit_a_json(edit), edit.created_at),
            )

    def _leer(self, row: Optional[sqlite3.Row]) -> Optional[PendingEdit]:
        if row is None:
            return None
        try:
            return _edit_desde_json(row["edit_json"])
        except (ValueError, KeyError, TypeError):
            # Fila de una versión vieja del formato o corrupta. Perder el
            # pendiente es malo; arrastrar la excepción hasta el handler y
            # dejar al usuario sin respuesta, peor.
            logger.warning("pedido pendiente ilegible para user=%s", row["user_id"])
            return None

    def get(self, user_id: int) -> Optional[PendingEdit]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pedidos_pendientes WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._leer(row)

    def pop(self, user_id: int) -> Optional[PendingEdit]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pedidos_pendientes WHERE user_id = ?", (user_id,)
            ).fetchone()
            conn.execute("DELETE FROM pedidos_pendientes WHERE user_id = ?", (user_id,))
        return self._leer(row)

    def has(self, user_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM pedidos_pendientes WHERE user_id = ?", (user_id,)
            ).fetchone() is not None

    def pop_expired(self, ttl_seconds: float) -> list[tuple[int, PendingEdit]]:
        corte = time.time() - ttl_seconds
        with self._connect() as conn:
            filas = conn.execute(
                "SELECT * FROM pedidos_pendientes WHERE created_at < ?", (corte,)
            ).fetchall()
            conn.execute("DELETE FROM pedidos_pendientes WHERE created_at < ?", (corte,))
        vencidos = []
        for row in filas:
            edit = self._leer(row)
            if edit is not None:
                vencidos.append((int(row["user_id"]), edit))
        return vencidos


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
                    poll_attempts INTEGER NOT NULL DEFAULT 0,
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
            if "poll_attempts" not in columns:
                # Va aparte de las de arriba: es INTEGER, no TEXT.
                conn.execute(
                    "ALTER TABLE trabajos_monitoreados "
                    "ADD COLUMN poll_attempts INTEGER NOT NULL DEFAULT 0"
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
                poll_attempts=r["poll_attempts"],
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

    def record_unreachable(self, job_id: str, owner_id: int) -> int:
        """Incrementa y devuelve la racha, en una sola ida a la base.

        `RETURNING` necesita SQLite 3.35 (2021). El fallback lee después
        del UPDATE: no es atómico, pero el único escritor de esta columna
        es el monitor, que corre de a un tick por vez.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE trabajos_monitoreados SET poll_attempts = poll_attempts + 1 "
                "WHERE job_id = ? AND owner_id = ?",
                (job_id, owner_id),
            )
            row = conn.execute(
                "SELECT poll_attempts FROM trabajos_monitoreados "
                "WHERE job_id = ? AND owner_id = ?",
                (job_id, owner_id),
            ).fetchone()
        return row["poll_attempts"] if row is not None else 0

    def clear_unreachable(self, job_id: str, owner_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trabajos_monitoreados SET poll_attempts = 0 "
                "WHERE job_id = ? AND owner_id = ? AND poll_attempts != 0",
                (job_id, owner_id),
            )
