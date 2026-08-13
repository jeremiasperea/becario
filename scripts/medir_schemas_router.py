#!/usr/bin/env python3
"""¿Conviene partir el router en dos etapas (clasificador + extractor)?

La idea que motiva este harness: en vez de una sola llamada con el schema
grande (11 intents), hacer una primera pasada que decida la FAMILIA del
pedido (sistema / cálculo) y una segunda con un schema chico y un prompt
afinado para esa familia. Suena bien y tiene precedente en el propio
código: `extract_structure()` ya es exactamente eso para el eje del
material, y su comentario explica que se agregó porque *todos* los modelos
locales sueltan `formula` cuando el mensaje trae otro eje de extracción.

Este script mide si esa intuición se sostiene, en vez de discutirla. Tres
brazos sobre los MISMOS pedidos reales de la bitácora (`chat_messages`):

  A  «schema grande»   `_SYSTEM_PROMPT` (6454 ch) + schema del plan (5027 ch).
                       Se parsea SIN backfill a propósito: interesa medir lo
                       que extrae el schema grande solo, no lo que produce la
                       red de seguridad que ya existe.
  B  «schema chico»    prompt enfocado + schema de la familia, con el MISMO
                       vocabulario que hoy. Es la propuesta implementada tal
                       cual: menos superficie, nada nuevo que decir.
  C  «lo que faltaba»  el schema chico MÁS la pieza de vocabulario que no
                       existía. Cambia por familia:
                         · sistema → enum `base` (home / corridas / absoluta)
                         · cálculo → el `_STRUCT_PROMPT` DE PRODUCCIÓN, que es
                           la segunda pasada especializada que ya está viva.

B contra A responde «¿alcanza con achicar la superficie?».
C contra B responde «¿el problema era que faltaba cómo decirlo?».

Ojo con leer A como si fuera producción: hoy producción es A + C combinados
(`route()` llama al backfill). A solo es el piso, no el comportamiento actual.

Métricas, distintas por familia porque los ejes de falla son distintos:

  sistema  ALUCINACIÓN de ruta — el modelo emite una ruta absoluta que no
           está escrita en el mensaje. Es literalmente lo que produjo
           `/home/ana`, `/` y `/Zr/bcc` en la bitácora. Y, para el brazo C,
           si además acertó el ancla.
  cálculo  FÓRMULA correcta, incluido el caso de control donde lo correcto
           es NO inventar ninguna. Es el eje que el código ya documenta como
           frágil en mensajes con geometría (losas).

Pega contra un Ollama REAL, así que corre fuera de pytest —`pyproject.toml`
limita `testpaths` a `tests/`— y además exige `BECARIO_LIVE_ROUTER_CHECK=1`,
el mismo cinturón que `live_router_check.py`.

Uso:
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/medir_schemas_router.py
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/medir_schemas_router.py \
        --familia calculo --repeticiones 5
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/medir_schemas_router.py \
        --modelo gemma4:12b --json docs/medicion_schemas.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.config import Settings  # noqa: E402
from becario.domain.models import RouterUnavailableError  # noqa: E402
from becario.infrastructure.ollama_router import (  # noqa: E402
    _STRUCT_PROMPT,
    _SYSTEM_PROMPT,
    OllamaRouter,
)

# ---------------------------------------------------------------------------
# Brazo B — sistema: schema chico, mismo vocabulario que hoy
# ---------------------------------------------------------------------------
# Deliberadamente NO tiene forma de decir «el home»: hoy tampoco la hay
# (`_resolve_workspace_path` solo distingue absoluta de relativa-a-corridas).
# Si el brazo B alucina rutas, no es porque el prompt esté mal escrito: es
# porque le pedimos algo que su vocabulario no puede expresar.
_B_SISTEMA_PROMPT = (
    "Sos el extractor de operaciones de archivos de B.E.C.A.R.I.O., un "
    "asistente de cluster HPC. El mensaje pide UNA operación sobre archivos o "
    "carpetas del cluster. Extraé la lista de operaciones, en orden.\n"
    "accion: 'listar' (ver qué hay), 'crear_carpeta', o 'ver_archivo'.\n"
    "ruta: la ruta que el usuario escribió. Si el usuario NO escribió una "
    "ruta explícita, dejala VACÍA. Nunca inventes una ruta.\n"
    "nombre_archivo: solo para ver_archivo, cuando el usuario nombra el "
    "archivo sin ruta (p. ej. CONTCAR, OSZICAR)."
)

_B_SISTEMA_SCHEMA = {
    "type": "object",
    "properties": {
        "operaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "accion": {
                        "type": "string",
                        "enum": ["listar", "crear_carpeta", "ver_archivo"],
                    },
                    "ruta": {"type": "string"},
                    "nombre_archivo": {"type": "string"},
                },
                "required": ["accion"],
            },
        }
    },
    "required": ["operaciones"],
}

# ---------------------------------------------------------------------------
# Brazo C — sistema: el mismo schema MÁS el ancla explícita
# ---------------------------------------------------------------------------
# `base` es el `RemotePath` del plan de refactor (D1) expresado en el schema:
# las tres formas que el dominio necesita distinguir, y ninguna más.
_C_SISTEMA_PROMPT = (
    "Sos el extractor de operaciones de archivos de B.E.C.A.R.I.O., un "
    "asistente de cluster HPC. El mensaje pide UNA operación sobre archivos o "
    "carpetas del cluster. Extraé la lista de operaciones, en orden.\n"
    "accion: 'listar' (ver qué hay), 'crear_carpeta', o 'ver_archivo'.\n"
    "base: DÓNDE ancla la ruta.\n"
    "  'home'     -> el usuario habla de SU home ('mi home', 'mi carpeta personal')\n"
    "  'corridas' -> el directorio de trabajo del bot ('la carpeta de corridas',\n"
    "                'mis cálculos'), y también cuando no dice dónde\n"
    "  'absoluta' -> el usuario escribió una ruta que empieza con /\n"
    "ruta: lo que va DESPUÉS de la base, sin barra inicial. Vacía si el "
    "usuario no nombró ninguna subcarpeta. Nunca inventes una ruta.\n"
    "nombre_archivo: solo para ver_archivo, cuando el usuario nombra el "
    "archivo sin ruta (p. ej. CONTCAR, OSZICAR)."
)

_C_SISTEMA_SCHEMA = {
    "type": "object",
    "properties": {
        "operaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "accion": {
                        "type": "string",
                        "enum": ["listar", "crear_carpeta", "ver_archivo"],
                    },
                    "base": {
                        "type": "string",
                        "enum": ["home", "corridas", "absoluta"],
                    },
                    "ruta": {"type": "string"},
                    "nombre_archivo": {"type": "string"},
                },
                "required": ["accion", "base"],
            },
        }
    },
    "required": ["operaciones"],
}

# ---------------------------------------------------------------------------
# Brazo B — cálculo: schema chico de la familia
# ---------------------------------------------------------------------------
# Incluye la geometría (miller/capas/supercelda) a propósito: el punto del
# experimento es justamente si `formula` sobrevive cuando compite con OTRO
# eje de extracción. Un schema chico que además le saque la geometría no
# probaría nada — ganaría por no tener rival.
_B_CALCULO_PROMPT = (
    "Sos el extractor de pedidos de cálculo de B.E.C.A.R.I.O., un asistente "
    "HPC de DFT/VASP. El mensaje pide un cálculo sobre una estructura atómica "
    "o generar el archivo de una estructura. Extraé:\n"
    "formula: símbolo o fórmula química (zirconio->Zr, tungsteno->W, "
    "silicio->Si). Si el mensaje no nombra ningún material, dejala VACÍA — "
    "no inventes uno.\n"
    "red_cristalina: bcc, fcc, hcp, diamond… solo si el mensaje la dice.\n"
    "tipo_calculo: relajacion, estatico, dos, convergencia_encut.\n"
    "tipo_estructura: bulk, slab, molecula.\n"
    "miller / capas / supercelda: solo para losas y superceldas."
)

_B_CALCULO_SCHEMA = {
    "type": "object",
    "properties": {
        "formula": {"type": "string"},
        "red_cristalina": {"type": "string"},
        "tipo_calculo": {"type": "string"},
        "tipo_estructura": {"type": "string"},
        "miller": {"type": "array", "items": {"type": "integer"}},
        "capas": {"type": "integer"},
        "supercelda": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["formula"],
}


# ---------------------------------------------------------------------------
# Casos: pedidos reales de la bitácora, con su origen anotado
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Caso:
    texto: str
    familia: str  # "sistema" | "calculo"
    origen: str
    # sistema
    base: Optional[str] = None
    # cálculo: "" significa «lo correcto es NO extraer ninguna»
    formula: Optional[str] = None
    red: Optional[str] = None


CASOS: tuple[Caso, ...] = (
    # ---------------- sistema ----------------
    Caso("listá mi home", "sistema",
         "chat_messages 9,11,13,17 — devolvió /home/ana", base="home"),
    Caso("stá mi home", "sistema",
         "chat_messages 15 — mismo pedido con un tipeo", base="home"),
    Caso("mostrame la estructura de archivos en el cluster", "sistema",
         "chat_messages 23,26,38", base="corridas"),
    Caso("mostramelo en forma de tree", "sistema",
         "chat_messages 32 — devolvió '/' y rebotó por ruta inválida",
         base="corridas"),
    Caso("quiero que crees una carpeta en la carpeta de corridas, llamada Zr",
         "sistema", "chat_messages 21 — devolvió /home/becario_runs/Zr",
         base="corridas"),
    Caso("crea una carpeta en la carpeta de corridas, llamada W", "sistema",
         "chat_messages 34", base="corridas"),
    Caso("dentro de Zr quiero crees 3 carpetas bcc, fcc y hcp", "sistema",
         "chat_messages 24 — creó /Zr/bcc en la raíz del cluster",
         base="corridas"),
    Caso("dentro de W quiero que crees 3 carpetas bcc, fcc y hcp", "sistema",
         "chat_messages 36", base="corridas"),
    Caso("Quiero ver el CONTCAR del último cálculo", "sistema",
         "chat_messages 1,3,5 — por nombre, sin ruta", base="corridas"),
    Caso("qué archivos hay en /data/becario_runs", "sistema",
         "control: ruta absoluta explícita, no puede romperse", base="absoluta"),

    # ---------------- cálculo ----------------
    Caso("relajá el bulk de W", "calculo",
         "chat_messages 37 y fixture del router", formula="W"),
    Caso("relajá el bulk de Zr hcp", "calculo",
         "chat_messages 93 — el único envío que llegó a sbatch",
         formula="Zr", red="hcp"),
    Caso("relajá el ZrO2", "calculo",
         "chat_messages 82,88,98,102,109 — 6 intentos", formula="ZrO2"),
    Caso("Armá un slab de ZrO2 (001) de 5 capas, 2x1", "calculo",
         "chat_messages 70,72,74,76 — el caso donde el código documenta "
         "que TODOS los modelos sueltan formula", formula="ZrO2"),
    Caso("Hacé la curva de convergencia de ENCUT para Zr hcp", "calculo",
         "ejemplo del HELP_TEXT que el bot ofrece", formula="Zr", red="hcp"),
    Caso("Generá un POSCAR de Si diamond 2x2x2", "calculo",
         "ejemplo del HELP_TEXT que el bot ofrece", formula="Si", red="diamond"),
    Caso("dame la densidad de estados del W bcc", "calculo",
         "tipo_calculo=dos, documentado en el README", formula="W", red="bcc"),
    Caso("relajá el bulk", "calculo",
         "control: sin material nombrado, NO se puede inventar uno",
         formula=""),
)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def alucina_ruta(ruta: str, texto: str) -> bool:
    """Ruta absoluta que NO está escrita en el mensaje del usuario.

    El criterio es la sustring literal, no un parseo: si el usuario escribió
    `/data/becario_runs`, emitirlo es citar; si escribió «mi home» y sale
    `/home/ana`, es inventar. Ese es exactamente el fallo de la bitácora.
    """
    ruta = (ruta or "").strip()
    if not ruta.startswith("/"):
        return False
    return ruta not in texto


def _norm(v) -> str:
    return str(v or "").strip().lower()


@dataclass
class Marca:
    """Lo que se le cuenta a un brazo en una repetición."""

    fallo: bool
    ancla_ok: bool = False
    detalle: str = ""


def evaluar_sistema(caso: Caso, ops: list[dict], con_base: bool) -> Marca:
    rutas = [str(o.get("ruta", "") or "") for o in ops]
    inventadas = [r for r in rutas if alucina_ruta(r, caso.texto)]
    ancla_ok = False
    if con_base and ops:
        ancla_ok = {_norm(o.get("base")) for o in ops} == {_norm(caso.base)}
    return Marca(
        fallo=bool(inventadas),
        ancla_ok=ancla_ok,
        detalle=f"inventó {inventadas[0]!r}" if inventadas else "",
    )


def evaluar_calculo(caso: Caso, params: dict) -> Marca:
    obtenida = _norm(params.get("formula"))
    esperada = _norm(caso.formula)
    if esperada == "":
        # Control: acertar es NO haber inventado nada.
        return Marca(
            fallo=bool(obtenida),
            detalle=f"inventó formula={obtenida!r}" if obtenida else "",
        )
    if obtenida != esperada:
        return Marca(
            fallo=True,
            detalle=f"formula={obtenida or '(vacía)'!r}, se esperaba {esperada!r}",
        )
    return Marca(fallo=False)


# ---------------------------------------------------------------------------
# Brazos
# ---------------------------------------------------------------------------
def brazo_a(router: OllamaRouter, caso: Caso):
    """Schema grande, UNA pasada, sin backfill.

    Se evita `route()` a propósito: ese método ya aplica `_backfill_structure`,
    que es el brazo C de la familia cálculo. Medirlos juntos haría que A se
    llevara el crédito de C.
    """
    t0 = time.monotonic()
    # Un fallo de infraestructura cuenta como brazo sin resultado, igual
    # que antes contaba el `None`: la medición son varias decenas de
    # minutos y no se tira por un timeout suelto.
    try:
        raw = router._chat(_SYSTEM_PROMPT, caso.texto, router._schema)
    except RouterUnavailableError:
        return [], {}, time.monotonic() - t0
    dt = time.monotonic() - t0
    plan = router.parse_llm_output(raw)
    ops = [
        {"accion": s.action.value,
         "ruta": s.parametros.get("destino_remoto", "") or "",
         "nombre_archivo": s.parametros.get("nombre_archivo", "") or ""}
        for s in plan.steps
    ]
    # Para cálculo interesa el primer paso que trae material.
    params: dict = {}
    for s in plan.steps:
        if s.parametros.get("formula"):
            params = dict(s.parametros)
            break
    return ops, params, dt


def brazo_chico(router: OllamaRouter, caso: Caso, prompt: str, schema: dict):
    t0 = time.monotonic()
    try:
        raw = router._chat(prompt, caso.texto, schema)
    except RouterUnavailableError:
        return [], {}, time.monotonic() - t0
    dt = time.monotonic() - t0
    if not raw:
        return [], {}, dt
    try:
        data = json.loads(raw)
    except ValueError:
        return [], {}, dt
    if caso.familia == "sistema":
        return data.get("operaciones", []), {}, dt
    return [], data, dt


def brazo_c_calculo(router: OllamaRouter, caso: Caso):
    """La segunda pasada DE PRODUCCIÓN (`_STRUCT_PROMPT`, 600 ch).

    No es una reconstrucción para el experimento: es el mismo código que hoy
    rescata `formula` cuando el schema grande la suelta.
    """
    t0 = time.monotonic()
    params = router.extract_structure(caso.texto)
    return [], params, time.monotonic() - t0


def correr_brazo(router: OllamaRouter, caso: Caso, brazo: str):
    if brazo == "A":
        return brazo_a(router, caso)
    if caso.familia == "sistema":
        prompt = _B_SISTEMA_PROMPT if brazo == "B" else _C_SISTEMA_PROMPT
        schema = _B_SISTEMA_SCHEMA if brazo == "B" else _C_SISTEMA_SCHEMA
        return brazo_chico(router, caso, prompt, schema)
    if brazo == "B":
        return brazo_chico(router, caso, _B_CALCULO_PROMPT, _B_CALCULO_SCHEMA)
    return brazo_c_calculo(router, caso)


def muestra(caso: Caso, ops: list[dict], params: dict) -> str:
    if caso.familia == "sistema":
        return str([(o.get("accion"), o.get("base", "-"),
                     o.get("ruta") or o.get("nombre_archivo", ""))
                    for o in ops])
    return str({k: params.get(k) for k in ("formula", "red_cristalina")
                if params.get(k)}) or "{}"


# ---------------------------------------------------------------------------
@dataclass
class Acumulado:
    fallos: int = 0
    anclas: int = 0
    total: int = 0
    lat: list[float] = field(default_factory=list)


def main() -> int:
    if os.environ.get("BECARIO_LIVE_ROUTER_CHECK") != "1":
        print(
            "Este harness pega contra un Ollama real y tarda varios minutos.\n"
            "Corrélo a propósito:\n\n"
            "    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python "
            "scripts/medir_schemas_router.py\n"
        )
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--familia", default="todas",
                        choices=("todas", "sistema", "calculo"))
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--modelo", default="", help="pisa BECARIO_OLLAMA_MODEL")
    parser.add_argument("--json", default="", help="archivo donde volcar el detalle")
    args = parser.parse_args()

    st = Settings.from_env()
    modelo = args.modelo or st.ollama_model
    router = OllamaRouter(base_url=st.ollama_url, model=modelo,
                          timeout=st.ollama_timeout_seconds)

    casos = [c for c in CASOS
             if args.familia == "todas" or c.familia == args.familia]

    print(f"modelo: {modelo}  ·  {len(casos)} caso(s) × {args.repeticiones} "
          f"repetición(es) × 3 brazos")
    # La primera generación paga la carga del modelo a RAM y ensuciaría la
    # latencia del primer caso (mismo criterio que live_router_check.py).
    print("warm-up…", flush=True)
    try:
        router._chat(_STRUCT_PROMPT, "relajá el bulk de W", router._params_schema)
    except RouterUnavailableError as exc:
        # El warm-up no mide nada; si falla, los brazos lo van a decir.
        print(f"  (warm-up falló: {exc.reason.value})")
    print()

    acum = {(f, b): Acumulado()
            for f in ("sistema", "calculo") for b in ("A", "B", "C")}
    detalle = []

    for caso in casos:
        print(f"── [{caso.familia}] {caso.texto!r}")
        print(f"   ({caso.origen})")
        fila = {"texto": caso.texto, "familia": caso.familia,
                "origen": caso.origen, "brazos": {}}
        for brazo in ("A", "B", "C"):
            a = acum[(caso.familia, brazo)]
            fallos = anclas = 0
            lats: list[float] = []
            primera = ""
            motivos: list[str] = []
            for i in range(args.repeticiones):
                ops, params, dt = correr_brazo(router, caso, brazo)
                lats.append(dt)
                if caso.familia == "sistema":
                    m = evaluar_sistema(caso, ops, con_base=(brazo == "C"))
                else:
                    m = evaluar_calculo(caso, params)
                fallos += int(m.fallo)
                anclas += int(m.ancla_ok)
                if m.detalle and m.detalle not in motivos:
                    motivos.append(m.detalle)
                if i == 0:
                    primera = muestra(caso, ops, params)
            a.fallos += fallos
            a.anclas += anclas
            a.total += args.repeticiones
            a.lat.extend(lats)
            marca = "🔴" if fallos else "🟢"
            extra = ""
            if caso.familia == "sistema" and brazo == "C":
                extra = f" ancla={anclas}/{args.repeticiones}"
            print(f"   {brazo} {marca} falló {fallos}/{args.repeticiones}{extra}  "
                  f"{statistics.median(lats):5.1f}s   {primera}")
            for m in motivos[:2]:
                print(f"        · {m}")
            fila["brazos"][brazo] = {
                "fallos": fallos, "anclas": anclas,
                "latencia_mediana": round(statistics.median(lats), 2),
                "muestra": primera, "motivos": motivos,
            }
        detalle.append(fila)
        print()

    print("=" * 74)
    for familia in ("sistema", "calculo"):
        if not any(c.familia == familia for c in casos):
            continue
        eje = "alucinaciones de ruta" if familia == "sistema" else "fórmulas erradas"
        print(f"\n{familia.upper()}  ({eje})")
        print(f"  {'brazo':6} {'fallos':>12} {'ancla ok':>11} {'latencia med':>14}")
        for brazo in ("A", "B", "C"):
            a = acum[(familia, brazo)]
            if not a.total:
                continue
            anc = (f"{a.anclas}/{a.total}"
                   if familia == "sistema" and brazo == "C" else "n/a")
            print(f"  {brazo:6} {a.fallos:>5}/{a.total:<6} {anc:>11} "
                  f"{statistics.median(a.lat):>13.1f}s")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "modelo": modelo,
            "fecha": datetime.now(timezone.utc).isoformat(),
            "repeticiones": args.repeticiones,
            "resumen": {
                f"{f}/{b}": {"fallos": a.fallos, "anclas": a.anclas,
                             "total": a.total,
                             "latencia_mediana": round(statistics.median(a.lat), 2)}
                for (f, b), a in acum.items() if a.total
            },
            "detalle": detalle,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ndetalle: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
