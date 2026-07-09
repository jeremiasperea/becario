"""Composition root de B.E.C.A.R.I.O.

Único lugar del proyecto que conoce todas las implementaciones concretas
y las cablea entre sí (Dependency Injection manual — no hace falta más).

Uso:
    export BECARIO_BOT_TOKEN=...
    export BECARIO_SSH_HOST=cluster.ejemplo.edu.ar
    export BECARIO_USERS_FILE=users.json   # roster del grupo, ver README

Para agregar/quitar miembros del grupo (no se hace por Telegram):
    python scripts/manage_users.py add --telegram-id 111111111 \
        --ssh-user jperez --ssh-key ~/.ssh/id_jperez --name "Juan Pérez"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from becario.application.job_monitor import JobMonitorService
from becario.application.services import BecarioService
from becario.config import ConfigError, Settings, required_config_present
from becario.infrastructure.ase_builder import ASEStructureBuilder
from becario.infrastructure.ollama_router import OllamaRouter
from becario.infrastructure.ssh_gateway import SSHClusterGatewayFactory
from becario.infrastructure.storage import (
    InMemoryConfirmationStore,
    SQLiteHistoryRepository,
    SQLiteJobTracker,
)
from becario.infrastructure.user_registry import JSONUserRegistry
from becario.presentation.telegram_bot import TelegramBot
from becario.setup_wizard import run_setup_wizard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def build_bot(settings: Settings) -> TelegramBot:
    router = OllamaRouter(base_url=settings.ollama_url, model=settings.ollama_model)
    registry = JSONUserRegistry(settings.users_file)
    cluster_factory = SSHClusterGatewayFactory(
        default_host=settings.ssh_host, default_port=settings.ssh_port
    )
    structures = ASEStructureBuilder(workdir=settings.structures_dir)
    history = SQLiteHistoryRepository(settings.db_path)
    history.ensure_schema()
    job_tracker = SQLiteJobTracker(settings.db_path)
    confirmations = InMemoryConfirmationStore(
        ttl_seconds=settings.confirmation_ttl_seconds
    )
    service = BecarioService(
        router=router,
        registry=registry,
        cluster_factory=cluster_factory,
        history=history,
        confirmations=confirmations,
        structures=structures,
        job_tracker=job_tracker,
    )
    job_monitor = JobMonitorService(
        registry=registry,
        cluster_factory=cluster_factory,
        tracker=job_tracker,
        history=history,
    )
    return TelegramBot(
        token=settings.telegram_token,
        service=service,
        job_monitor=job_monitor,
        monitor_interval_seconds=settings.monitor_interval_seconds,
    )


def _roster_is_empty(users_file: str) -> bool:
    """True si todavía no hay ninguna cuenta del cluster registrada."""
    path = Path(users_file)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return True
    return not data.get("users")


def main() -> None:
    # 1) Si falta la config obligatoria, guiar al usuario para crearla.
    if not required_config_present():
        run_setup_wizard()

    # 2) Cargar la config (con mensaje amigable si algo quedó mal).
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"\n❌ Problema con la configuración: {exc}")
        print(f"   Revisá o borrá el archivo .env y volvé a correr el programa.")
        raise SystemExit(1)

    # 3) Aviso amigable si todavía no registró su cuenta del cluster.
    if _roster_is_empty(settings.users_file):
        print(
            "\n⚠️  Todavía no registraste tu cuenta del cluster, así que el bot\n"
            "    no va a poder ejecutar nada hasta que lo hagas. Corré:\n\n"
            "        python scripts/manage_users.py add\n"
        )

    # 4) Arrancar el bot, traduciendo errores de arranque a mensajes claros.
    try:
        build_bot(settings).run()
    except Exception as exc:  # noqa: BLE001 - queremos un mensaje amigable, no un traceback
        name = type(exc).__name__
        if name == "InvalidToken":
            print(
                "\n❌ El token del bot de Telegram no es válido.\n"
                "   Verificá el token con @BotFather, borrá el archivo .env y volvé a correr."
            )
        elif name in ("NetworkError", "TimedOut"):
            print(
                "\n❌ No pude conectarme a Telegram. Revisá tu conexión a internet\n"
                "   y volvé a intentar."
            )
        else:
            raise
        raise SystemExit(1)


if __name__ == "__main__":
    main()
