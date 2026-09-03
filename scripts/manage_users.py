#!/usr/bin/env python3
"""CLI para administrar el roster de B.E.C.A.R.I.O. (users.json).

No hay administración desde Telegram a propósito: agregar o quitar gente
del cluster es una decisión que se toma fuera del chat.

Uso (interactivo, recomendado para gente sin experiencia):
    python3 scripts/manage_users.py add   # te pregunta los datos uno por uno

Uso (con flags, para automatizar):
    python3 scripts/manage_users.py add --telegram-id 111111111 \
        --ssh-user jperez --ssh-key /home/becario/.ssh/id_jperez \
        --name "Juan Pérez"
    python3 scripts/manage_users.py list
    python3 scripts/manage_users.py remove --telegram-id 111111111
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
    """Roster leído con la MISMA tolerancia que quien lo consume.

    `JSONUserRegistry.reload()` hace `raw.get("users", [])`: un archivo sin
    esa clave lo carga como roster vacío y el bot arranca sin drama. Este
    CLI hacía `data["users"]` directo y moría con un `KeyError` pelado
    sobre exactamente el mismo archivo — o sea que no se podía administrar
    un roster que el bot considera válido, justo el día en que hace falta
    arreglarlo.

    Un JSON ROTO sigue explotando, y es a propósito (ver
    `test_un_json_corrupto_explota_en_vez_de_devolver_un_roster_vacio`):
    tratarlo como vacío haría que la siguiente alta lo reescriba con una
    sola persona y se lleve puesto al grupo. La diferencia es que `{}` no
    es un archivo roto — no hay nada que perder ahí.
    """
    if not path.exists():
        return {"users": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"❌ {path} no es un objeto JSON; no lo toco.")
        raise SystemExit(1)
    usuarios = data.get("users") or []
    if not isinstance(usuarios, list):
        # Una clave `users` que no es lista sí es un archivo mal formado:
        # normalizarla a `[]` en silencio perdería lo que hubiera adentro.
        print(f"❌ La clave 'users' de {path} no es una lista; no lo toco.")
        raise SystemExit(1)
    data["users"] = usuarios
    return data


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

    # Un flag pasado VACÍO es un error, no un dato ausente. `argparse` ya
    # los distingue: `None` es "no lo pasaron", `""` es "lo pasaron vacío".
    # Confundirlos mandaba a preguntar por teclado, así que un script
    # automatizado con una variable sin expandir se quedaba esperando stdin
    # en vez de fallar con código 1 — un cron colgado en vez de un error.
    for flag, valor in (("--ssh-user", args.ssh_user), ("--ssh-key", args.ssh_key)):
        if valor is not None and not valor.strip():
            print(f"❌ {flag} vino vacío. Omitilo para que te lo pregunte, "
                  f"o pasale un valor.")
            raise SystemExit(1)

    # Modo interactivo: si falta algún dato por flag, se pide por teclado.
    interactive = (
        args.telegram_id is None or args.ssh_user is None or args.ssh_key is None
    )
    if interactive:
        print("Registrar tu cuenta del cluster (Enter acepta el valor sugerido).\n")

    telegram_id = args.telegram_id
    if telegram_id is None:
        print("Tu id numérico de Telegram (te lo dice el bot @userinfobot).")
        telegram_id = _prompt_int("  id de Telegram")

    ssh_user = (
        args.ssh_user if args.ssh_user is not None
        else _prompt("Tu usuario SSH en el cluster")
    )
    ssh_key = (
        args.ssh_key if args.ssh_key is not None
        else _prompt("Ruta a tu clave privada SSH", str(Path("~/.ssh/id_rsa")))
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

    # `.get` y no `[...]`: una entrada a la que le falte el id no puede
    # tumbar el alta de OTRA persona. Al no matchear, además, sobrevive al
    # filtro — borrar en silencio una entrada que no entendemos sería peor
    # que dejarla; `list` la muestra marcada para que alguien la arregle.
    data["users"] = [
        u for u in data["users"]
        if u.get("telegram_user_id") != identity.telegram_user_id
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
        # Una entrada incompleta se MUESTRA marcada en vez de tumbar el
        # listado entero: `list` es la herramienta con la que alguien va a
        # diagnosticar por qué el bot no lo reconoce, y el registro descarta
        # esa entrada logueando el error. Explotar acá dejaría al que
        # administra sin forma de ver qué está mal.
        falta = [c for c in ("telegram_user_id", "ssh_user") if not u.get(c)]
        marca = f"  ⚠️ entrada inválida, le falta: {', '.join(falta)}" if falta else ""
        print(
            f"- {u.get('display_name') or '(sin nombre)'}: "
            f"telegram_id={u.get('telegram_user_id', '?')} "
            f"ssh_user={u.get('ssh_user', '?')}{marca}"
        )


def cmd_remove(args: argparse.Namespace) -> None:
    """Quita a alguien del roster. Si no había nada que quitar, no escribe.

    El `_save` incondicional era un arma cargada: sobre un `--file` con un
    dedazo (o antes de la primera alta) `_load` devolvía `{"users": []}` y
    el guardado lo materializaba, dejando un roster vacío nuevo en disco.
    Quien lo corrió creyó haber dado de baja a alguien y en realidad creó
    un archivo que no existía. No escribir cuando no cambió nada también
    evita tocarle la fecha de modificación a un archivo de control de
    acceso sin motivo.
    """
    path = Path(args.file)
    data = _load(path)
    quedan = [u for u in data["users"] if u.get("telegram_user_id") != args.telegram_id]
    if len(quedan) == len(data["users"]):
        print(f"⚠️ No encontrado: {args.telegram_id}")
        return
    data["users"] = quedan
    _save(path, data)
    print("✅ Eliminado")


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
