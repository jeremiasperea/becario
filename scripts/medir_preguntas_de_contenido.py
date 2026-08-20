#!/usr/bin/env python3
"""¿E6 pide INTERPRETAR un archivo, o alcanza con parsearlo y saber rutearlo?

El caso que lo motiva, de la sesión real del 2026-08-20:

    vos> Muéstrame el OSZICAR
    bot> 📄 …/OSZICAR: [el archivo, truncado]
    vos> Cuantas vueltas iónicas hizo?
    bot> ❓ No pude interpretar tu pedido.

`plan_lo_que_mostro_usarlo.md` lo anota como «el más grande de los seis»
porque «requiere que el bot interprete un archivo, no que lo muestre», y
dice —bien— que conviene medirlo antes de diseñarlo. Esto es esa medición.

La hipótesis que se pone a prueba es que el diagnóstico está corrido de
lugar: que estas preguntas no piden interpretación sino un PARSEO, que
buena parte de esos parseos ya existen en el repo, y que lo que falta es
poder decirlo — o sea, el mismo bug que E1, un escalón más adentro.

Dos brazos sobre el mismo corpus:

  A  «inventario»   Sin LLM y sin red. Para cada pregunta: qué HECHO pide,
                    de qué archivo sale, y si el repo ya lo calcula en
                    algún lado. Lo que ya se calcula se verifica corriendo
                    ese código sobre un fragmento real, no afirmando que
                    anda. Contesta: ¿cuánto de E6 es código que falta?

  B  «ruteo»        Con Ollama vivo. Qué emite `route()` hoy para cada
                    pregunta. Contesta: ¿cuánto de E6 es vocabulario que
                    falta? El síntoma medido —una pregunta que termina en
                    el texto de ayuda— vive acá.

Sobre el corpus: hay UNA pregunta real, la de arriba. Una no es una
medición, así que se completa con preguntas de la misma familia sobre los
archivos que el bot efectivamente muestra. Van marcadas `sintetica` y se
reportan aparte: mezclarlas con la real haría parecer medido lo que es
imaginado, que es justo lo que este plan vino a evitar.

El brazo B pega contra un Ollama REAL, así que corre fuera de pytest
—`pyproject.toml` limita `testpaths` a `tests/`— y exige
`BECARIO_LIVE_ROUTER_CHECK=1`, el mismo cinturón que `live_router_check.py`.

Uso:
    .venv/bin/python scripts/medir_preguntas_de_contenido.py
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python \
        scripts/medir_preguntas_de_contenido.py --ruteo
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python \
        scripts/medir_preguntas_de_contenido.py --ruteo \
        --modelo qwen2.5-coder:14b --json docs/medicion_contenido.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.application.relaxed_source import (  # noqa: E402
    _IONIC_STEP_RE,
    _NSW_RE,
)
from becario.application.handlers.queries import (  # noqa: E402
    _parse_last_e0,
    _parse_poscar_cell,
)
from becario.domain.models import RouterUnavailableError  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _ROOT / "tests" / "fixtures" / "contenido"

# El OSZICAR REAL de la corrida sobre la que se preguntó (job 14,
# Zr_relajacion), tal como el bot lo mostró en el chat — o sea TRUNCADO,
# que es parte del hallazgo: ni el humano podía contar sobre lo que vio.
_OSZICAR_REAL = _FIXTURES / "oszicar_zr_relajacion_truncado.txt"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pregunta:
    """Una pregunta sobre el contenido de un archivo de la corrida.

    `ya_existe` es la parte cara de esta medición y la que no se puede
    automatizar: sale de leer el repo y anotar DÓNDE está hoy el cálculo
    del hecho que la pregunta pide. Vacío significa que no está.
    """

    texto: str
    origen: str          # "real" | "sintetica"
    archivo: str         # de qué archivo sale el hecho
    hecho: str           # qué hay que calcular
    ya_existe: str = ""  # dónde lo calcula el repo hoy
    # Verificador: corre el código que ya existe sobre el fragmento real y
    # devuelve el valor. `None` = no hay nada que correr todavía.
    verificar: Optional[Callable[[], object]] = field(default=None, compare=False)


def _pasos_ionicos() -> int:
    return len(_IONIC_STEP_RE.findall(_OSZICAR_REAL.read_text(encoding="utf-8")))


def _ultima_energia() -> Optional[float]:
    return _parse_last_e0(_OSZICAR_REAL.read_text(encoding="utf-8"))


CORPUS: list[Pregunta] = [
    Pregunta(
        texto="Cuantas vueltas iónicas hizo?",
        origen="real",
        archivo="OSZICAR",
        hecho="contar los pasos iónicos (líneas 'N F= …')",
        ya_existe="relaxed_source._IONIC_STEP_RE (se usa en _check_convergence)",
        verificar=_pasos_ionicos,
    ),
    Pregunta(
        texto="qué energía dio?",
        origen="sintetica",
        archivo="OSZICAR",
        hecho="último E0",
        ya_existe="queries._parse_last_e0 y job_monitor._parse_last_e0",
        verificar=_ultima_energia,
    ),
    Pregunta(
        texto="convergió la relajación?",
        origen="sintetica",
        archivo="OSZICAR + INCAR",
        hecho="pasos iónicos contra NSW",
        ya_existe="relaxed_source._check_convergence (devuelve además el aviso)",
    ),
    Pregunta(
        texto="cuántos átomos tiene la celda?",
        origen="sintetica",
        archivo="POSCAR/CONTCAR",
        hecho="suma de la línea de conteos",
        ya_existe="job_monitor._parse_poscar_n_atoms",
    ),
    Pregunta(
        texto="cuánto le quedó el parámetro de red?",
        origen="sintetica",
        archivo="CONTCAR",
        hecho="a, b, c, α, β, γ de la celda",
        ya_existe="queries._parse_poscar_cell (es lo que contesta consultar_resultados)",
    ),
    Pregunta(
        texto="con qué ENCUT corrió?",
        origen="sintetica",
        archivo="INCAR",
        hecho="leer un tag del INCAR",
        ya_existe="",  # `_NSW_RE` lee UNO; no hay lector genérico de tags
    ),
    Pregunta(
        texto="cuántos pasos electrónicos tardó la primera vuelta?",
        origen="sintetica",
        archivo="OSZICAR",
        hecho="contar líneas DAV/RMM antes del primer 'F='",
        ya_existe="",
    ),
    Pregunta(
        texto="por qué falló?",
        origen="sintetica",
        archivo="vasp.out + código de salida",
        hecho="diagnóstico del fallo",
        ya_existe="job_monitor (el aviso de trabajo terminado ya lo arma)",
    ),
    Pregunta(
        texto="resumime qué dice el OUTCAR",
        origen="sintetica",
        archivo="OUTCAR",
        hecho="resumen libre",
        ya_existe="",  # esta SÍ es interpretación: no hay hecho puntual
    ),
]


# ---------------------------------------------------------------------------
# Brazo A — inventario (sin LLM)
# ---------------------------------------------------------------------------
def inventario() -> dict:
    """Cuánto de E6 es código que falta, y cuánto ya está escrito."""
    filas = []
    for p in CORPUS:
        valor, error = None, None
        if p.verificar is not None:
            try:
                valor = p.verificar()
            except Exception as exc:  # el fragmento podría no estar
                error = f"{type(exc).__name__}: {exc}"
        filas.append({
            "pregunta": p.texto,
            "origen": p.origen,
            "archivo": p.archivo,
            "hecho": p.hecho,
            "ya_existe": p.ya_existe,
            "valor_verificado": valor,
            "error": error,
        })
    con_codigo = [f for f in filas if f["ya_existe"]]
    return {
        "total": len(filas),
        "ya_calculado_en_el_repo": len(con_codigo),
        "filas": filas,
    }


def imprimir_inventario(inv: dict) -> None:
    print("== Brazo A — inventario (sin LLM) ==")
    for f in inv["filas"]:
        marca = "✅" if f["ya_existe"] else "  "
        etiqueta = "REAL " if f["origen"] == "real" else "sint."
        print(f"{marca} [{etiqueta}] {f['pregunta']}")
        print(f"       {f['archivo']} · {f['hecho']}")
        if f["ya_existe"]:
            print(f"       ya está en: {f['ya_existe']}")
        if f["valor_verificado"] is not None:
            print(f"       verificado sobre el fragmento real -> {f['valor_verificado']}")
        if f["error"]:
            print(f"       ⚠️ {f['error']}")
    n, total = inv["ya_calculado_en_el_repo"], inv["total"]
    print(f"   ── {n}/{total} ya se calculan en el repo, en otro handler")


# ---------------------------------------------------------------------------
# Brazo B — ruteo (con Ollama vivo)
# ---------------------------------------------------------------------------
# La primera versión de este brazo contaba «ruteos que nombran el archivo» y
# daba 24/27, que sonaba bien y no medía nada: el router manda casi todo a
# `consultar_resultados`, que contesta UNA cosa fija —parámetros de red y E0
# de la última corrida— sin importar qué se preguntó. Un plan que llega al
# handler correcto y no lleva la pregunta adentro no sirve para contestarla.
#
# Lo que se mide entonces es si el plan DISTINGUE una pregunta de otra. Si
# «cuántas vueltas iónicas hizo» y «qué energía dio» producen el mismo plan
# byte a byte, el ruteo perdió el pedido, y da igual en qué handler cayó.


def medir_ruteo(modelo: str, url: str, repeticiones: int, timeout: float) -> dict:
    from becario.infrastructure.ollama_router import OllamaRouter

    router = OllamaRouter(model=modelo, base_url=url, timeout=timeout)
    filas = []
    for p in CORPUS:
        emitidos, latencias, fallos = [], [], 0
        for _ in range(repeticiones):
            arranque = time.monotonic()
            try:
                plan = router.route(p.texto)
            except RouterUnavailableError as exc:
                fallos += 1
                emitidos.append(f"<router caído: {exc.reason.value}>")
                continue
            finally:
                latencias.append(time.monotonic() - arranque)
            emitidos.append(_huella(plan))
        filas.append({
            "pregunta": p.texto,
            "origen": p.origen,
            "hecho": p.hecho,
            "emitidos": emitidos,
            "intentos": repeticiones,
            "fallos_de_router": fallos,
            "latencia_mediana": round(sorted(latencias)[len(latencias) // 2], 2),
        })
    return {"modelo": modelo, "repeticiones": repeticiones, "filas": filas}


def _huella(plan) -> str:
    """El plan emitido, como texto comparable: intents Y parámetros.

    Los parámetros van a propósito. Si el ruteo llevara la pregunta adentro
    —en un campo, en el nombre del archivo, en lo que sea— se vería acá.
    """
    return " + ".join(
        s.action.value + (f"({sorted(s.parametros.items())})" if s.parametros else "")
        for s in plan.steps
    )


def _hecho_de(pregunta: str) -> str:
    """El hecho que pide una pregunta, buscado en el corpus por su texto.

    Existe para que `--desde-json` sepa leer corridas guardadas antes de que
    el reporte incluyera el hecho: una medición vieja no debería perder
    contexto solo porque el formato creció.
    """
    return next((p.hecho for p in CORPUS if p.texto == pregunta), "?")


def imprimir_ruteo(res: dict) -> None:
    print(f"\n== Brazo B — ruteo con {res['modelo']} ==")
    # Cuántas preguntas comparten plan con otra: son las que el ruteo no
    # puede distinguir, o sea las que se van a contestar con otra cosa.
    por_huella: dict[str, list[str]] = {}
    for f in res["filas"]:
        for e in dict.fromkeys(f["emitidos"]):
            por_huella.setdefault(e, []).append(f["pregunta"])
    for f in res["filas"]:
        etiqueta = "REAL " if f["origen"] == "real" else "sint."
        huellas = list(dict.fromkeys(f["emitidos"]))
        choca = any(len(por_huella[h]) > 1 for h in huellas)
        print(f"{'❌' if choca else '✅'} [{etiqueta}] {f['pregunta']}  "
              f"({f['latencia_mediana']}s)")
        print(f"       hecho pedido: {f.get('hecho') or _hecho_de(f['pregunta'])}")
        for h in huellas:
            comparten = len(por_huella[h]) - 1
            sufijo = f"  ← igual que otras {comparten}" if comparten else ""
            print(f"       -> {h}{sufijo}")
    planes = len(por_huella)
    hechos = len({f.get("hecho") or _hecho_de(f["pregunta"]) for f in res["filas"]})
    colisionan = sum(
        1 for f in res["filas"]
        if any(len(por_huella[h]) > 1 for h in dict.fromkeys(f["emitidos"]))
    )
    print(f"   ── {planes} planes distintos para {hechos} hechos distintos; "
          f"{colisionan}/{len(res['filas'])} preguntas comparten plan con otra")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ruteo", action="store_true",
        help="además del inventario, medir qué emite el router (necesita Ollama)",
    )
    parser.add_argument("--modelo", default="qwen2.5-coder:14b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", metavar="PATH", default=None)
    parser.add_argument(
        "--desde-json", metavar="PATH", default=None,
        help="re-imprimir el brazo B desde una corrida guardada, sin tocar Ollama",
    )
    args = parser.parse_args()

    inv = inventario()
    imprimir_inventario(inv)

    reporte = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inventario": inv,
    }
    if args.desde_json:
        guardado = json.loads(Path(args.desde_json).read_text(encoding="utf-8"))
        imprimir_ruteo(guardado["ruteo"])
        return 0
    if args.ruteo:
        if os.environ.get("BECARIO_LIVE_ROUTER_CHECK") != "1":
            print(
                "\n--ruteo pega contra un Ollama real: exportá "
                "BECARIO_LIVE_ROUTER_CHECK=1 para confirmarlo explícitamente.",
                file=sys.stderr,
            )
            return 1
        res = medir_ruteo(args.modelo, args.url, args.repeticiones, args.timeout)
        imprimir_ruteo(res)
        reporte["ruteo"] = res

    if args.json:
        Path(args.json).write_text(
            json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nReporte escrito en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
