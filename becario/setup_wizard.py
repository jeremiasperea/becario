"""Asistente de primer arranque de B.E.C.A.R.I.O.

Cuando el programa arranca y no encuentra la configuración obligatoria, este
asistente la pide de forma interactiva (guiada y en español) y la guarda en un
archivo `.env` local que NO se sube a ningún repositorio (está en .gitignore).

Solo configura lo *global* (token del bot, host del cluster, Ollama). El alta de
tu cuenta del cluster se hace después con `scripts/manage_users.py`, que también
es interactivo.
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path

from becario.config import ENV_PATH


def _ask(prompt: str, default: str = "") -> str:
    """Pide un valor por teclado. Enter acepta el default si lo hay."""
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        print("  ⚠️  Este dato es obligatorio, por favor completalo.")


def _ask_secret(prompt: str) -> str:
    """Pide un secreto sin mostrarlo en pantalla."""
    while True:
        value = getpass.getpass(f"{prompt}: ").strip()
        if value:
            return value
        print("  ⚠️  Este dato es obligatorio, por favor completalo.")


def _write_env(path: Path, values: dict[str, str]) -> None:
    lines = [
        "# Configuración de B.E.C.A.R.I.O. generada por el asistente de arranque.",
        "# Archivo LOCAL con secretos: NO lo subas a ningún repositorio.",
        "",
    ]
    lines += [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # solo el dueño puede leer/escribir el archivo
    except OSError:
        pass  # en algunos sistemas de archivos (p.ej. Windows) chmod no aplica


def run_setup_wizard(env_path: Path = ENV_PATH) -> None:
    """Corre el asistente interactivo y escribe el archivo `.env`."""
    print("=" * 64)
    print("  Bienvenido/a a B.E.C.A.R.I.O. 🤖")
    print("=" * 64)
    print(
        "\nNo encontré la configuración, así que vamos a crearla juntos.\n"
        "Te voy a pedir unos pocos datos. Se guardan en un archivo local\n"
        f"({env_path.name}) que NO se sube a ningún repositorio.\n"
        "Podés apretar Enter para aceptar el valor sugerido entre [corchetes].\n"
    )

    print("1) Token del bot de Telegram")
    print("   Lo obtenés hablándole a @BotFather en Telegram y creando un bot.")
    token = _ask_secret("   Pegá el token (no se mostrará en pantalla)")

    print("\n2) Servidor del cluster HPC")
    print("   Es el host al que te conectás por SSH (ej: cluster.ejemplo.edu.ar).")
    ssh_host = _ask("   Host del cluster")
    ssh_port = _ask("   Puerto SSH", "22")

    print("\n3) Ollama (el modelo de lenguaje local que interpreta los mensajes)")
    print("   Si no sabés qué poner, dejá los valores por defecto.")
    ollama_url = _ask("   URL de Ollama", "http://localhost:11434")
    ollama_model = _ask("   Modelo de Ollama", "gemma4:12b")

    values = {
        "BECARIO_BOT_TOKEN": token,
        "BECARIO_SSH_HOST": ssh_host,
        "BECARIO_SSH_PORT": ssh_port,
        "BECARIO_OLLAMA_URL": ollama_url,
        "BECARIO_OLLAMA_MODEL": ollama_model,
    }

    print("\n4) Cálculos VASP (opcional: podés configurarlo después en el .env)")
    print("   Para que el bot prepare cálculos completos necesita saber dónde")
    print("   está la biblioteca de POTCAR en el cluster (un subdirectorio por")
    print("   elemento, p. ej. /data/potcars/Zr_sv/POTCAR) y cómo correr VASP.")
    potcar_dir = input(
        "   Ruta de la biblioteca de POTCAR (Enter para saltear): "
    ).strip()
    if potcar_dir:
        values["BECARIO_POTCAR_DIR"] = potcar_dir
        values["BECARIO_VASP_CMD"] = _ask(
            "   Comando para correr VASP", "mpirun vasp_std"
        )
        vasp_prelude = input(
            "   Línea previa del script (módulos/exports, Enter para ninguna): "
        ).strip()
        if vasp_prelude:
            values["BECARIO_VASP_PRELUDE"] = vasp_prelude
    _write_env(env_path, values)

    # Dejar la config disponible en este mismo proceso.
    for key, value in values.items():
        os.environ.setdefault(key, value)

    print("\n" + "=" * 64)
    print(f"✅ Listo. Guardé tu configuración en {env_path}")
    print("=" * 64)
    print(
        "\nÚltimo paso antes de usar el bot: registrá tu cuenta del cluster con\n\n"
        "    python3 scripts/manage_users.py add\n\n"
        "(te va a pedir tu id de Telegram, tu usuario SSH y tu clave privada).\n"
    )
