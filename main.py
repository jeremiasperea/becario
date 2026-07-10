"""Composition root de B.E.C.A.R.I.O.

Único lugar del proyecto que conoce todas las implementaciones concretas
y las cablea entre sí (Dependency Injection manual — no hace falta más).

Uso:
    python3 main.py

La primera vez, si no encuentra configuración, un asistente interactivo pide
token del bot, host del cluster y Ollama, y los guarda en un `.env` local
(gitignoreado). El entorno tiene prioridad sobre el `.env` si preferís exportar
las variables a mano.

Para registrar tu cuenta del cluster (no se hace por Telegram):
    python3 scripts/manage_users.py add
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
# httpx loguea la URL completa en INFO, y esa URL incluye el token del bot.
# Lo subimos a WARNING para no filtrar el token a los logs (aplica también a
# las llamadas internas de python-telegram-bot).
logging.getLogger("httpx").setLevel(logging.WARNING)


def build_bot(settings: Settings) -> TelegramBot:
    router = OllamaRouter(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
    )
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


def _check_telegram_token(token: str, timeout: float = 10.0) -> None:
    """Valida el token contra la API de Telegram ANTES de arrancar el polling.

    python-telegram-bot no propaga el error de token inválido fuera de
    run_polling() (lo maneja en su retry loop interno y lo loguea con un
    traceback), así que hacemos una verificación temprana con un mensaje claro.
    """
    import httpx

    try:
        resp = httpx.get(
            f"https://api.telegram.org/bot{token}/getMe", timeout=timeout
        )
    except httpx.HTTPError:
        print(
            "\n❌ No pude conectarme a Telegram para verificar el token.\n"
            "   Revisá tu conexión a internet y volvé a intentar."
        )
        raise SystemExit(1)

    if resp.status_code == 200 and resp.json().get("ok"):
        return  # token válido
    if resp.status_code in (401, 404):
        print(
            "\n❌ El token del bot de Telegram no es válido.\n"
            "   Verificá el token con @BotFather, borrá el archivo .env y volvé a correr."
        )
        raise SystemExit(1)
    print(
        f"\n❌ Telegram respondió algo inesperado (HTTP {resp.status_code}) al "
        "verificar el token.\n   Volvé a intentar en un rato."
    )
    raise SystemExit(1)


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
            "        python3 scripts/manage_users.py add\n"
        )

    # 4) Verificar el token temprano (mensaje claro en vez de un traceback de PTB).
    _check_telegram_token(settings.telegram_token)

    # 5) Arrancar el bot. El token ya fue validado; una caída de red posterior
    #    la maneja PTB con su propio backoff.
    build_bot(settings).run()


if __name__ == "__main__":
    main()
