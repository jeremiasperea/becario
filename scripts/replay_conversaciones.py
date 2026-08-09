"""Batería de conversaciones: replay del chat real contra el sistema vivo.

Reproduce, contra el cluster de prueba + Ollama + Materials Project reales,
los pedidos que un usuario efectivamente le hizo al bot por Telegram (la
tabla `chat_messages` de `becario.db`), y comprueba que la respuesta sea la
que corresponde.

Por qué existe, si ya hay `scripts/live_router_check.py`: ese mide el ROUTER
(texto → pasos). Acá se mide el sistema COMPLETO — router, handlers, SSH,
ASE, Materials Project, POTCARs, confirmaciones y estado pendiente — que es
donde aparecieron casi todas las fallas que se ven en la bitácora. Un plan
correcto que después pierde la fórmula entre paso y paso pasa el chequeo del
router y le arruina la tarde al usuario igual.

Los escenarios viven en `tests/conversaciones/*.txt`, en un formato de línea
legible (ver `parse_scenario`). Cada uno es una conversación completa y
autocontenida: los mensajes que solo tienen sentido como respuesta a una
repregunta («tetragonal») incluyen el turno que la provoca.

Uso:
    .venv/bin/python scripts/replay_conversaciones.py
    .venv/bin/python scripts/replay_conversaciones.py --solo CV17,CV18
    .venv/bin/python scripts/replay_conversaciones.py --repeticiones 3
    .venv/bin/python scripts/replay_conversaciones.py --json informe.json

Aislamiento (importante): por defecto NO toca la base ni el directorio de
corridas de producción. Usa una copia temporal de la base y un `remote_base`
propio bajo `/data/becario_qa/<timestamp>`, así el replay no ensucia la
bitácora que le da origen ni pisa corridas reales. Con `--sin-aislar` se corre
contra lo que diga el `.env` (útil para reproducir un caso puntual a mano).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# El transcriptor de voz baja el modelo de Whisper al construirse y acá se
# escribe, no se dicta. El entorno gana sobre el `.env`.
os.environ["BECARIO_WHISPER_MODEL"] = "off"

_SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "tests" / "conversaciones"


# ---------------------------------------------------------------------------
# Formato de escenario
# ---------------------------------------------------------------------------
#   # comentario
#   id: CV07
#   titulo: ...
#   origen: chat_messages 32,40,42
#   sintoma: ...          (qué se vio fallar en la bitácora; solo documenta)
#   > texto del usuario   (turno: mensaje entrante)
#   boton: confirmar|cancelar|modificar
#   espera: ok|error|confirmacion|dato   (flags esperados del último turno)
#   contiene: substring
#   no_contiene: substring
#   regex: patrón
#   no_regex: patrón
#
# Las aserciones aplican al ÚLTIMO turno declarado. `{base}` en una aserción
# se reemplaza por el remote_base efectivo y `{home}` por el home remoto.


@dataclass
class Assertion:
    kind: str
    value: str

    def check(self, reply_text: str, flags: set[str], subs: dict[str, str]) -> Optional[str]:
        value = self.value
        for key, repl in subs.items():
            value = value.replace("{" + key + "}", repl)
        if self.kind == "contiene":
            if value.lower() not in reply_text.lower():
                return f"falta el texto {value!r}"
        elif self.kind == "no_contiene":
            if value.lower() in reply_text.lower():
                return f"aparece el texto prohibido {value!r}"
        elif self.kind == "regex":
            if not re.search(value, reply_text, re.IGNORECASE | re.DOTALL):
                return f"no matchea /{value}/"
        elif self.kind == "no_regex":
            if re.search(value, reply_text, re.IGNORECASE | re.DOTALL):
                return f"matchea el patrón prohibido /{value}/"
        elif self.kind == "espera":
            faltan = {f.strip() for f in value.split(",") if f.strip()} - flags
            if faltan:
                return f"faltan los estados {sorted(faltan)} (hubo: {sorted(flags)})"
        return None


@dataclass
class Turn:
    kind: str  # "texto" | "boton"
    value: str
    assertions: list[Assertion] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    titulo: str
    origen: str
    sintoma: str
    turns: list[Turn]
    path: Path


def parse_scenario(text: str, path: Path) -> Scenario:
    meta: dict[str, str] = {}
    turns: list[Turn] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            turns.append(Turn(kind="texto", value=line[1:].strip()))
            continue
        if line.lower() == "reiniciar":
            # Turno sin mensaje: reconstruye el servicio, como un
            # `systemctl restart` en medio de la conversación. Es la única
            # forma de probar que el estado conversacional sobrevive, y lo
            # que la batería no podía expresar cuando vivía en memoria.
            turns.append(Turn(kind="reiniciar", value=""))
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{path.name}: línea sin 'clave:' ni '>': {raw!r}")
        key, value = key.strip().lower(), value.strip()
        if key == "boton":
            turns.append(Turn(kind="boton", value=value.lower()))
        elif key in ("contiene", "no_contiene", "regex", "no_regex", "espera"):
            if not turns:
                raise ValueError(f"{path.name}: aserción {key!r} antes del primer turno")
            turns[-1].assertions.append(Assertion(kind=key, value=value))
        else:
            meta[key] = value
    if "id" not in meta:
        raise ValueError(f"{path.name}: falta 'id:'")
    if not turns:
        raise ValueError(f"{path.name}: no tiene ningún turno")
    return Scenario(
        id=meta["id"],
        titulo=meta.get("titulo", ""),
        origen=meta.get("origen", ""),
        sintoma=meta.get("sintoma", ""),
        turns=turns,
        path=path,
    )


def load_scenarios(directory: Path = _SCENARIOS_DIR) -> list[Scenario]:
    escenarios = [parse_scenario(p.read_text(encoding="utf-8"), p)
                  for p in sorted(directory.glob("*.txt"))]
    vistos: dict[str, Path] = {}
    for sc in escenarios:
        if sc.id in vistos:
            raise ValueError(f"id duplicado {sc.id!r}: {sc.path.name} y {vistos[sc.id].name}")
        vistos[sc.id] = sc.path
    return escenarios


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    entrada: str
    kind: str
    respuesta: str
    flags: list[str]
    fallas: list[str]
    latencia: float

    @property
    def ok(self) -> bool:
        return not self.fallas


@dataclass
class RunResult:
    scenario_id: str
    turns: list[TurnResult]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(t.ok for t in self.turns)

    @property
    def fallas(self) -> list[str]:
        if self.error:
            return [f"excepción: {self.error}"]
        salida = []
        for t in self.turns:
            salida.extend(f"[{t.entrada[:48]!r}] {f}" for f in t.fallas)
        return salida


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def _flatten(reply) -> tuple[str, set[str], Optional[str]]:
    """Aplana `reply` + sus followups como los ve el usuario en el chat.

    Devuelve (texto completo, flags, token de confirmación más reciente).
    El canal manda la respuesta y después cada followup como mensajes
    separados; para las aserciones de contenido eso es un solo bloque.
    """
    partes: list[str] = []
    token: Optional[str] = None
    flags: set[str] = set()

    def walk(r, top: bool) -> None:
        nonlocal token
        partes.append(r.text)
        if r.needs_confirmation and r.confirmation_token:
            token = r.confirmation_token
        if r.needs_confirmation:
            flags.add("confirmacion")
        if r.awaiting_params:
            flags.add("dato")
        if top:
            flags.add("ok" if r.ok else "error")
        for f in r.followups:
            walk(f, top=False)

    walk(reply, top=True)
    return "\n".join(partes), flags, token


def run_scenario(sc: Scenario, build_service, user_id: int, chat_id: int,
                 subs: dict[str, str]) -> RunResult:
    """Corre un escenario sobre un servicio recién construido.

    Servicio nuevo por escenario a propósito: los pendientes y las
    confirmaciones viven en memoria por usuario, y compartirlos haría que
    una conversación contamine a la siguiente — justo el tipo de acople que
    la batería tiene que poder detectar, no producir.
    """
    turns: list[TurnResult] = []
    try:
        service = build_service()
    except Exception:
        return RunResult(scenario_id=sc.id, turns=[], error=traceback.format_exc(limit=3))

    token: Optional[str] = None
    for turn in sc.turns:
        started = time.monotonic()
        try:
            if turn.kind == "reiniciar":
                # El token y el pendiente NO se tocan: si están en disco
                # siguen ahí, y si estaban en memoria acaban de morir —
                # que es exactamente lo que se quiere detectar.
                service = build_service()
                turns.append(TurnResult(
                    entrada="(reinicio del bot)", kind=turn.kind, respuesta="",
                    flags=[], fallas=[], latencia=time.monotonic() - started,
                ))
                continue
            if turn.kind == "texto":
                reply = service.handle_text(chat_id=chat_id, user_id=user_id, text=turn.value)
            elif turn.value == "confirmar":
                if token is None:
                    raise RuntimeError("el escenario aprieta ✅ sin una confirmación viva")
                reply = service.confirm(token, requester_id=user_id)
            elif turn.value == "cancelar":
                if token is None:
                    raise RuntimeError("el escenario aprieta ❌ sin una confirmación viva")
                reply = service.reject(token, requester_id=user_id)
            elif turn.value == "modificar":
                # Token vencido/inexistente incluido a propósito: hay
                # escenarios que prueban justamente ese mensaje.
                reply = service.start_modification(
                    token or "token-inexistente", requester_id=user_id, chat_id=chat_id
                )
            else:
                raise ValueError(f"botón desconocido: {turn.value!r}")
        except Exception:
            return RunResult(
                scenario_id=sc.id, turns=turns, error=traceback.format_exc(limit=4)
            )
        latencia = time.monotonic() - started
        texto, flags, nuevo_token = _flatten(reply)
        if nuevo_token:
            token = nuevo_token
        fallas = [f for f in (a.check(texto, flags, subs) for a in turn.assertions) if f]
        turns.append(TurnResult(
            entrada=turn.value, kind=turn.kind, respuesta=texto,
            flags=sorted(flags), fallas=fallas, latencia=latencia,
        ))
    return RunResult(scenario_id=sc.id, turns=turns)


# ---------------------------------------------------------------------------
# Entorno de prueba
# ---------------------------------------------------------------------------
def _isolate(settings, remote_base: Optional[str]):
    """Copia de la base y `remote_base` propio, para no ensuciar producción."""
    from dataclasses import replace

    tmpdir = Path(tempfile.mkdtemp(prefix="becario-replay-"))
    db = tmpdir / "replay.db"
    origen = Path(settings.db_path)
    if origen.exists():
        # Se copia (en vez de arrancar vacía) para que los escenarios que
        # miran el historial encuentren las corridas que ya existen.
        shutil.copy2(origen, db)
    base = remote_base or (
        settings.remote_base.rstrip("/") + "_qa/"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    return replace(settings, db_path=str(db), remote_base=base), tmpdir, base


def _remote_home(settings) -> str:
    """Home real de la cuenta del roster, para las aserciones `{home}`."""
    from becario.infrastructure.ssh_gateway import SSHClusterGatewayFactory
    from becario.infrastructure.user_registry import JSONUserRegistry

    registry = JSONUserRegistry(settings.users_file)
    roster = json.loads(Path(settings.users_file).read_text(encoding="utf-8"))
    entries = roster["users"] if isinstance(roster, dict) else roster
    identity = registry.get_identity(int(entries[0]["telegram_user_id"]))
    factory = SSHClusterGatewayFactory(
        default_host=settings.ssh_host, default_port=settings.ssh_port
    )
    gateway = factory.for_identity(identity)
    for name in ("home", "remote_home", "get_home"):
        fn = getattr(gateway, name, None)
        if callable(fn):
            try:
                return str(fn()).strip()
            except Exception:
                pass
    return f"/home/{identity.ssh_user}" if identity.ssh_user != "root" else "/root"


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo", default="", help="ids separados por coma (CV03,CV17)")
    parser.add_argument("--repeticiones", type=int, default=1,
                        help="corridas por escenario; >1 mide inestabilidad del LLM")
    parser.add_argument("--json", default="", help="archivo donde volcar el informe")
    parser.add_argument("--remote-base", default="", help="directorio de corridas a usar")
    parser.add_argument("--sin-aislar", action="store_true",
                        help="usar la base y el remote_base del .env (ensucia producción)")
    parser.add_argument("--modelo", default="", help="pisa BECARIO_OLLAMA_MODEL")
    parser.add_argument("--verboso", action="store_true", help="imprime cada respuesta")
    args = parser.parse_args()

    if args.modelo:
        os.environ["BECARIO_OLLAMA_MODEL"] = args.modelo

    from becario.config import Settings
    from main import build_bot

    settings = Settings.from_env()
    tmpdir = None
    if args.sin_aislar:
        base = settings.remote_base
    else:
        settings, tmpdir, base = _isolate(settings, args.remote_base or None)

    roster = json.loads(Path(settings.users_file).read_text(encoding="utf-8"))
    entries = roster["users"] if isinstance(roster, dict) else roster
    user_id = int(entries[0]["telegram_user_id"])

    try:
        home = _remote_home(settings)
    except Exception as exc:  # el cluster puede estar caído: se avisa y sigue
        print(f"⚠️  No pude resolver el home remoto ({exc}); uso '/root'.")
        home = "/root"
    subs = {"base": base, "home": home}

    escenarios = load_scenarios()
    if args.solo:
        pedidos = {s.strip().upper() for s in args.solo.split(",") if s.strip()}
        escenarios = [s for s in escenarios if s.id.upper() in pedidos]
        if not escenarios:
            print(f"❌ Ningún escenario matchea {sorted(pedidos)}")
            return 2

    print(f"modelo: {settings.ollama_model}  ·  usuario: {user_id}")
    print(f"corridas en: {base}  ·  base de datos: {settings.db_path}")
    print(f"home remoto: {home}")
    print(f"{len(escenarios)} escenario(s) × {args.repeticiones} repetición(es)\n")

    def build_service():
        return build_bot(settings)._service

    informe: list[dict] = []
    t0 = time.monotonic()
    for i, sc in enumerate(escenarios, 1):
        corridas: list[RunResult] = []
        for rep in range(args.repeticiones):
            # chat_id propio por corrida: dos repeticiones no comparten hilo.
            res = run_scenario(sc, build_service, user_id, chat_id=900000 + i * 10 + rep,
                               subs=subs)
            corridas.append(res)
        exitos = sum(1 for c in corridas if c.ok)
        latencias = [t.latencia for c in corridas for t in c.turns]
        if exitos == args.repeticiones:
            marca = "✅"
        elif exitos == 0:
            marca = "❌"
        else:
            marca = "🟡"
        print(f"{marca} {sc.id}  {exitos}/{args.repeticiones}  "
              f"({statistics.median(latencias):.1f}s/turno)  {sc.titulo}")
        if exitos < args.repeticiones:
            vistas: list[str] = []
            for c in corridas:
                for f in c.fallas:
                    if f not in vistas:
                        vistas.append(f)
            for f in vistas[:6]:
                print(f"      · {f}")
        if args.verboso:
            for t in corridas[0].turns:
                print(f"      vos> {t.entrada}")
                for line in t.respuesta.splitlines():
                    print(f"      bot> {line}")
        informe.append({
            "id": sc.id, "titulo": sc.titulo, "origen": sc.origen,
            "sintoma": sc.sintoma, "exitos": exitos, "corridas": args.repeticiones,
            "fallas": sorted({f for c in corridas for f in c.fallas}),
            "turnos": [
                {"entrada": t.entrada, "tipo": t.kind, "flags": t.flags,
                 "latencia": round(t.latencia, 2), "fallas": t.fallas,
                 "respuesta": t.respuesta}
                for t in corridas[0].turns
            ],
        })

    total = len(escenarios)
    verdes = sum(1 for r in informe if r["exitos"] == args.repeticiones)
    print(f"\n{verdes}/{total} escenarios verdes en {time.monotonic() - t0:.0f}s")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "modelo": settings.ollama_model, "remote_base": base,
            "fecha": datetime.now(timezone.utc).isoformat(),
            "repeticiones": args.repeticiones, "escenarios": informe,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"informe: {args.json}")

    if tmpdir is not None:
        print(f"(base temporal en {tmpdir}, borrala cuando no la necesites)")
    return 0 if verdes == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
