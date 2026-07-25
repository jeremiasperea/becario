#!/usr/bin/env python3
"""Launcher que se asegura de que Ollama esté arriba antes de arrancar el bot.

`main.py` valida Ollama y aborta si no está (fail-fast, ver ADR-1/ADR-2): esa
decisión no cambia. Este script vive una capa más afuera y es el que *remedia*
antes de arrancar: si el servidor local no responde lo levanta, y si falta el
modelo ofrece bajarlo. Después le pasa el control a `main.py`, que vuelve a
validar por su cuenta — el gate de verdad sigue siendo el de la app.

Uso:
    python3 scripts/start_becario.py           # interactivo
    python3 scripts/start_becario.py --yes     # sin preguntas (systemd, CI)
    python3 scripts/start_becario.py --timeout 90

Solo levanta Ollama si `BECARIO_OLLAMA_URL` apunta a esta máquina: contra un
servidor remoto no hay nada que arrancar localmente, así que avisa y corta.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.config import ENV_PATH, PROJECT_ROOT, load_local_env
from becario.infrastructure.ollama_router import (
    OllamaModelMissingError,
    OllamaRouter,
    OllamaUnreachableError,
)

# Hosts que consideramos "esta máquina". Si la URL apunta a otro lado, no hay
# nada que levantar localmente y el script no intenta adivinar.
LOCAL_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1", "0.0.0.0"})

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:12b"

# Log del `ollama serve` que arranca este script (gitignoreado). Sin esto, si
# el servidor muere al arrancar no queda rastro de por qué.
OLLAMA_LOG = PROJECT_ROOT / "ollama.log"

# Registro público de Ollama: el manifest de un modelo lista sus capas con el
# tamaño en bytes, así podemos decir cuánto pesa ANTES de empezar a bajarlo.
MODEL_REGISTRY = "https://registry.ollama.ai/v2"

# Instalador oficial de Ollama (https://ollama.com/download). Pide sudo por su
# cuenta. Nunca se corre sin que el usuario lo apruebe viendo el comando exacto.
INSTALL_COMMAND = "curl -fsSL https://ollama.com/install.sh | sh"

# Margen sobre el peso del modelo por debajo del cual avisamos: bajar algo que
# te deja el disco al filo es tan malo como no poder bajarlo.
DISK_MARGIN = 1.10


def is_local_url(url: str) -> bool:
    """True si la URL apunta a esta máquina (único caso donde podemos arrancar)."""
    return (urlsplit(url).hostname or "") in LOCAL_HOSTS


def ollama_host_env(url: str) -> str:
    """`host:puerto` para OLLAMA_HOST, respetando un puerto no estándar.

    Si el `.env` apunta a http://localhost:11500, el servidor tiene que bindear
    ahí y no en el 11434 por defecto.
    """
    parts = urlsplit(url)
    return f"{parts.hostname or '127.0.0.1'}:{parts.port or 11434}"


def human_size(num_bytes: int) -> str:
    """Bytes a GB/MB decimales, que es como los muestra el propio Ollama."""
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.1f} GB"
    return f"{num_bytes / 1_000_000:.0f} MB"


def manifest_path(model: str) -> str:
    """`qwen2.5:7b` -> `library/qwen2.5/manifests/7b`.

    Sin tag, Ollama asume `latest`; sin namespace, la biblioteca oficial.
    """
    name, _, tag = model.partition(":")
    if "/" not in name:
        name = f"library/{name}"
    return f"{name}/manifests/{tag or 'latest'}"


def model_download_size(model: str, *, timeout: float = 10.0) -> int | None:
    """Peso de la descarga en bytes, o None si no se pudo averiguar.

    Es *best-effort* a propósito: si el registro no contesta seguimos adelante
    sin el dato en vez de bloquear el arranque por un chequeo informativo.
    """
    try:
        response = httpx.get(
            f"{MODEL_REGISTRY}/{manifest_path(model)}",
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        manifest = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    layers = manifest.get("layers") or []
    config = manifest.get("config") or {}
    try:
        return sum(int(layer["size"]) for layer in layers) + int(config.get("size", 0))
    except (KeyError, TypeError, ValueError):
        return None


def models_dir() -> Path:
    """Dónde va a escribir el modelo el servidor que levanta este script.

    Ojo: es el store del usuario que corre `ollama serve`, no un directorio
    global. Un Ollama de sistema (usuario `ollama`) usa otro distinto.
    """
    override = os.environ.get("OLLAMA_MODELS", "").strip()
    return Path(override) if override else Path.home() / ".ollama" / "models"


def free_space(path: Path) -> int | None:
    """Bytes libres en el filesystem de `path`, subiendo hasta un padre que exista."""
    current = path.resolve()
    while not current.exists():
        if current.parent == current:
            return None
        current = current.parent
    try:
        return shutil.disk_usage(current).free
    except OSError:
        return None


def _ask(prompt: str) -> bool:
    """Pregunta sí/no. Sin terminal (systemd, CI) devuelve False."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in ("s", "si", "sí", "y", "yes")


def ensure_binary(*, assume_yes: bool) -> None:
    """Se asegura de que exista el comando `ollama`, ofreciendo instalarlo."""
    if shutil.which("ollama"):
        return

    print("\n⚠️  Ollama no está instalado en esta máquina.")
    print(f"   El instalador oficial corre: {INSTALL_COMMAND}")
    print("   (te va a pedir sudo para copiar el binario y crear el servicio)")

    if not assume_yes and not _ask("   ¿Lo instalo ahora? [s/N] "):
        print(
            "   Ok, no instalo nada. Podés hacerlo a mano desde\n"
            "   https://ollama.com/download y volver a correr este script."
        )
        raise SystemExit(1)

    print("⏳ Instalando Ollama…")
    if subprocess.run(INSTALL_COMMAND, shell=True).returncode != 0:
        print("\n❌ Falló la instalación de Ollama. Probá a mano con:")
        print(f"   {INSTALL_COMMAND}")
        raise SystemExit(1)

    if not shutil.which("ollama"):
        print(
            "\n❌ El instalador terminó pero `ollama` sigue sin estar en el PATH.\n"
            "   Abrí una terminal nueva y volvé a correr este script."
        )
        raise SystemExit(1)
    print("✅ Ollama instalado.")


def probe(router: OllamaRouter, *, timeout: float = 2.0) -> str:
    """Estado del servidor: 'ok' | 'sin-modelo' | 'caido'.

    Reutiliza el chequeo del adaptador (`ensure_model_available`) en vez de
    reimplementar el HTTP acá: una sola definición de "Ollama está listo".
    """
    try:
        router.ensure_model_available(timeout=timeout)
    except OllamaUnreachableError:
        return "caido"
    except OllamaModelMissingError:
        return "sin-modelo"
    return "ok"


def wait_until_reachable(
    router: OllamaRouter, process: subprocess.Popen, *, deadline: float
) -> bool:
    """Espera a que el servidor responda, o a que el proceso se muera antes."""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False  # murió al arrancar; el caller muestra el log
        if probe(router) != "caido":
            return True
        time.sleep(0.5)
    return False


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])
    except OSError:
        return "(no pude leer el log)"


def start_ollama(
    url: str, router: OllamaRouter, *, timeout: float, assume_yes: bool = False
) -> None:
    """Arranca `ollama serve` en background y espera a que atienda."""
    ensure_binary(assume_yes=assume_yes)

    print(f"⏳ Ollama no responde en {url}. Levantando `ollama serve`…")

    env = {**os.environ, "OLLAMA_HOST": ollama_host_env(url)}
    with OLLAMA_LOG.open("ab") as log:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=log,
            stderr=subprocess.STDOUT,
            # Sesión propia: un Ctrl+C sobre el bot no se lleva puesto al servidor.
            start_new_session=True,
            env=env,
        )

    if not wait_until_reachable(router, process, deadline=time.monotonic() + timeout):
        print(
            f"\n❌ Arranqué `ollama serve` pero no atendió en {timeout:.0f}s.\n"
            f"   Últimas líneas de {OLLAMA_LOG}:\n\n{_tail(OLLAMA_LOG)}\n"
        )
        raise SystemExit(1)

    print(
        f"✅ Ollama arriba (PID {process.pid}). Queda corriendo en background;\n"
        f"   para pararlo: kill {process.pid}   (log en {OLLAMA_LOG})"
    )


def check_disk(model: str, size: int | None) -> None:
    """Corta si el modelo no entra en disco; avisa si entra muy justo.

    Este chequeo NO lo saltea `--yes`: esa flag aprueba la intención de bajar,
    no hace que el espacio aparezca. Mejor cortar acá que dejar el disco lleno
    a mitad de camino.
    """
    destino = models_dir()
    libre = free_space(destino)
    if libre is None:
        return
    print(f"   Espacio libre en {destino}: {human_size(libre)}")
    if size is None:
        return
    if libre < size:
        print(
            f"\n❌ No entra: el modelo pesa {human_size(size)} y te quedan "
            f"{human_size(libre)}.\n   Liberá espacio o elegí un modelo más chico "
            "(BECARIO_OLLAMA_MODEL)."
        )
        raise SystemExit(1)
    if libre < size * DISK_MARGIN:
        print(
            f"   ⚠️  Va a quedar muy justo: {human_size(libre - size)} libres "
            "después de bajarlo."
        )


def pull_model(model: str, *, assume_yes: bool) -> None:
    """Baja el modelo faltante, informando peso y espacio, y pidiendo permiso.

    Un pull puede ser de varios GB: no se dispara solo sin avisar.
    """
    ensure_binary(assume_yes=assume_yes)

    print(f"\n⚠️  El modelo {model!r} no está instalado en Ollama.")
    size = model_download_size(model)
    if size is None:
        print("   No pude consultar el peso en el registro de Ollama.")
    else:
        print(f"   Peso de la descarga: {human_size(size)}")

    check_disk(model, size)

    if not assume_yes and not _ask(
        f"   ¿Lo bajo ahora con `ollama pull {model}`? [s/N] "
    ):
        print(
            f"   Ok, no bajo nada. Instalalo con: ollama pull {model}\n"
            "   (o apuntá BECARIO_OLLAMA_MODEL a un modelo que ya tengas)"
        )
        raise SystemExit(1)

    print(f"⏳ Bajando {model}… (esto puede tardar)")
    if subprocess.run(["ollama", "pull", model]).returncode != 0:
        print(f"\n❌ Falló `ollama pull {model}`.")
        raise SystemExit(1)


def ensure_ollama(url: str, model: str, *, timeout: float, assume_yes: bool) -> None:
    """Deja Ollama en condiciones de atender, o corta con un mensaje claro."""
    router = OllamaRouter(base_url=url, model=model)
    state = probe(router, timeout=5.0)

    if state == "caido":
        if not is_local_url(url):
            print(
                f"\n❌ Ollama no responde en {url} y no es un servidor local:\n"
                "   no hay nada que yo pueda levantar desde acá.\n"
                "   Arrancalo en esa máquina o corregí BECARIO_OLLAMA_URL."
            )
            raise SystemExit(1)
        start_ollama(url, router, timeout=timeout, assume_yes=assume_yes)
        state = probe(router, timeout=5.0)

    if state == "sin-modelo":
        if not is_local_url(url):
            print(
                f"\n❌ El modelo {model!r} no está en el Ollama remoto de {url}.\n"
                f"   Instalalo allá con: ollama pull {model}"
            )
            raise SystemExit(1)
        pull_model(model, assume_yes=assume_yes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Levanta Ollama si hace falta y después arranca el bot."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="no preguntar nada (baja el modelo faltante sin confirmar)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="segundos a esperar a que Ollama atienda (default: 60)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="dejar Ollama listo pero no arrancar el bot",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Mismas fuentes y defaults que Settings.from_env(), pero sin construir
    # Settings: eso exige token y host SSH, y en el primer arranque todavía no
    # existen — el asistente de main.py es quien los pide.
    load_local_env(ENV_PATH)
    url = os.environ.get("BECARIO_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("BECARIO_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    ensure_ollama(url, model, timeout=args.timeout, assume_yes=args.yes)

    if args.check_only:
        print(f"✅ Ollama listo en {url} con el modelo {model!r}.")
        return

    # Le cedemos el proceso a main.py: un solo PID para el bot, y los señales
    # (Ctrl+C, systemd stop) le llegan directo sin intermediarios.
    os.execv(sys.executable, [sys.executable, str(PROJECT_ROOT / "main.py")])


if __name__ == "__main__":
    main()
