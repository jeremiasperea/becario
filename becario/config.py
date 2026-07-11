"""Configuración centralizada.

Las variables se leen del entorno; si no están en el entorno, se cargan desde
un archivo `.env` local (que NO se sube a git). Ese `.env` lo crea el asistente
de primer arranque (`becario/setup_wizard.py`) la primera vez que se ejecuta el
bot, así un usuario inexperto no tiene que exportar variables a mano.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Raíz del proyecto (dos niveles arriba de este archivo: becario/becario/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Variables obligatorias para poder arrancar.
REQUIRED_VARS = ("BECARIO_BOT_TOKEN", "BECARIO_SSH_HOST")


class ConfigError(RuntimeError):
    pass


def load_local_env(path: Path = ENV_PATH) -> None:
    """Carga un archivo `.env` estilo KEY=VALUE dentro de os.environ.

    - Ignora líneas en blanco y comentarios (`#`).
    - Soporta el prefijo `export ` (por si alguien copia comandos de shell).
    - Quita comillas simples o dobles que envuelvan el valor.
    - NO pisa variables que ya estén definidas en el entorno: el entorno gana
      sobre el archivo (útil para systemd/CI que inyectan variables reales).

    Sin dependencias externas a propósito (no usa python-dotenv).
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def required_config_present(path: Path = ENV_PATH) -> bool:
    """True si las variables obligatorias están disponibles (tras cargar el .env)."""
    load_local_env(path)
    return all(os.environ.get(name, "").strip() for name in REQUIRED_VARS)


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Falta la variable de entorno {name}")
    return value


def _int_env(name: str, default: str) -> int:
    raw = os.environ.get(name, default).strip() or default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} debe ser un número entero, no {raw!r}") from exc


def _float_env(name: str, default: str) -> float:
    raw = os.environ.get(name, default).strip() or default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} debe ser un número, no {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    ssh_host: str
    ssh_port: int = 22
    users_file: str = "users.json"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"
    ollama_timeout_seconds: float = 120.0
    db_path: str = "becario.db"
    structures_dir: str = "./structures"
    confirmation_ttl_seconds: float = 600.0
    monitor_interval_seconds: float = 60.0
    # Cálculos VASP: biblioteca de POTCAR en el cluster (ruta absoluta,
    # con un subdirectorio por elemento: Zr_sv/POTCAR, W/POTCAR…),
    # comando de ejecución y prelude del script de corrida.
    potcar_dir: str = ""
    vasp_cmd: str = "vasp_std"
    vasp_prelude: str = ""
    # Dónde viven las corridas en el cluster (relativa al home remoto
    # salvo que empiece con /).
    remote_base: str = "becario_runs"
    # Transcripción local de notas de voz (faster-whisper). "off" la apaga;
    # tiny/base/small/medium según la RAM disponible.
    whisper_model: str = "small"
    whisper_language: str = "es"

    @classmethod
    def from_env(cls) -> "Settings":
        load_local_env(ENV_PATH)
        return cls(
            telegram_token=_require("BECARIO_BOT_TOKEN"),
            ssh_host=_require("BECARIO_SSH_HOST"),
            ssh_port=_int_env("BECARIO_SSH_PORT", "22"),
            users_file=os.environ.get("BECARIO_USERS_FILE", "users.json"),
            ollama_url=os.environ.get("BECARIO_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.environ.get("BECARIO_OLLAMA_MODEL", "gemma4:12b"),
            ollama_timeout_seconds=_float_env("BECARIO_OLLAMA_TIMEOUT", "120"),
            db_path=os.environ.get("BECARIO_DB_PATH", "becario.db"),
            structures_dir=os.environ.get("BECARIO_STRUCTURES_DIR", "./structures"),
            confirmation_ttl_seconds=_float_env("BECARIO_CONFIRM_TTL", "600"),
            monitor_interval_seconds=_float_env("BECARIO_MONITOR_INTERVAL", "60"),
            potcar_dir=os.environ.get("BECARIO_POTCAR_DIR", "").strip(),
            vasp_cmd=os.environ.get("BECARIO_VASP_CMD", "vasp_std").strip() or "vasp_std",
            vasp_prelude=os.environ.get("BECARIO_VASP_PRELUDE", "").strip(),
            remote_base=os.environ.get("BECARIO_REMOTE_BASE", "becario_runs").strip()
            or "becario_runs",
            whisper_model=os.environ.get("BECARIO_WHISPER_MODEL", "small").strip()
            or "small",
            whisper_language=os.environ.get("BECARIO_WHISPER_LANGUAGE", "es").strip(),
        )
