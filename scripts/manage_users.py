#!/usr/bin/env python3
"""CLI para administrar el roster de B.E.C.A.R.I.O. (users.json).

No hay administración desde Telegram a propósito: agregar o quitar gente
del cluster es una decisión que se toma fuera del chat.

Uso (interactivo, recomendado para gente sin experiencia):
    python scripts/manage_users.py add   # te pregunta los datos uno por uno

Uso (con flags, para automatizar):
    python scripts/manage_users.py add --telegram-id 111111111 \
        --ssh-user jperez --ssh-key /home/becario/.ssh/id_jperez \
        --name "Juan Pérez"
    python scripts/manage_users.py list
    python scripts/manage_users.py remove --telegram-id 111111111
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.domain.models import ClusterIdentity  # noqa: E402
from pydantic import ValidationError  # noqa: E402


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"users": []}


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _prompt(prompt: str, default: str = "", required: bool = True) -> str:
    """Pide un valor por teclado. Enter acepta el default (o vacío si no es obligatorio)."""
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default or not required:
            return default
        print("  ⚠️  Este dato es obligatorio, por favor completalo.")


def _prompt_int(prompt: str) -> int:
    while True:
        raw = _prompt(prompt)
        try:
            return int(raw)
        except ValueError:
            print(f"  ⚠️  '{raw}' no es un número. Debe ser solo dígitos.")


def cmd_add(args: argparse.Namespace) -> None:
    path = Path(args.file)
    data = _load(path)

    # Modo interactivo: si falta algún dato por flag, se pide por teclado.
    interactive = args.telegram_id is None or not args.ssh_user or not args.ssh_key
    if interactive:
        print("Registrar tu cuenta del cluster (Enter acepta el valor sugerido).\n")

    telegram_id = args.telegram_id
    if telegram_id is None:
        print("Tu id numérico de Telegram (te lo dice el bot @userinfobot).")
        telegram_id = _prompt_int("  id de Telegram")

    ssh_user = args.ssh_user or _prompt("Tu usuario SSH en el cluster")
    ssh_key = args.ssh_key or _prompt(
        "Ruta a tu clave privada SSH", str(Path("~/.ssh/id_rsa"))
    )
    name = args.name if args.name else (
        _prompt("Tu nombre (opcional)", required=False) if interactive else ""
    )
    ssh_host = args.ssh_host
    if ssh_host is None and interactive:
        ssh_host = _prompt(
            "Host del cluster (Enter = usar el host global)", required=False
        ) or None

    try:
        identity = ClusterIdentity(
            telegram_user_id=telegram_id,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key,
            display_name=name or "",
            ssh_host=ssh_host,
        )
    except ValidationError as exc:
        print(f"❌ Datos inválidos:\n{exc}")
        raise SystemExit(1)

    if not Path(ssh_key).expanduser().exists():
        print(f"⚠️  Aviso: no encuentro la clave {ssh_key} en esta máquina "
              "(puede estar bien si el bot corre en otro servidor).")

    data["users"] = [
        u for u in data["users"] if u["telegram_user_id"] != identity.telegram_user_id
    ]
    data["users"].append(json.loads(identity.model_dump_json(exclude_none=True)))
    _save(path, data)
    print(f"✅ {identity.display_name or identity.ssh_user} "
          f"(telegram_id={identity.telegram_user_id}) agregado/actualizado en {path}")


def cmd_list(args: argparse.Namespace) -> None:
    data = _load(Path(args.file))
    if not data["users"]:
        print("Roster vacío.")
        return
    for u in data["users"]:
        print(
            f"- {u.get('display_name') or '(sin nombre)'}: "
            f"telegram_id={u['telegram_user_id']} ssh_user={u['ssh_user']}"
        )


def cmd_remove(args: argparse.Namespace) -> None:
    path = Path(args.file)
    data = _load(path)
    before = len(data["users"])
    data["users"] = [u for u in data["users"] if u["telegram_user_id"] != args.telegram_id]
    _save(path, data)
    removed = before - len(data["users"])
    print(f"✅ Eliminado" if removed else f"⚠️ No encontrado: {args.telegram_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Administrar el roster de B.E.C.A.R.I.O.")
    parser.add_argument("--file", default="users.json", help="Ruta al roster (default: users.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser(
        "add",
        help="Agregar o actualizar un miembro (si omitís datos, te los pregunta)",
    )
    p_add.add_argument("--telegram-id", type=int, default=None)
    p_add.add_argument("--ssh-user", default=None)
    p_add.add_argument("--ssh-key", default=None)
    p_add.add_argument("--name", default="")
    p_add.add_argument("--ssh-host", default=None, help="Solo si difiere del host global")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="Listar miembros")
    p_list.set_defaults(func=cmd_list)

    p_rm = sub.add_parser("remove", help="Quitar un miembro")
    p_rm.add_argument("--telegram-id", type=int, required=True)
    p_rm.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
