#!/usr/bin/env python3
"""¿Dónde está el techo REAL del schema del router?

El presupuesto anterior (4197 B = 3650 x 1.15) nunca fue una medición: el
3650 se midió antes de fusionar `steps` y el 15% era headroom para ESE
refactor (ADR-0006). Este script busca el número medido, y es el que hay
que volver a correr antes de tocar el techo de nuevo.

Método: inflar el schema con relleno SEMÁNTICAMENTE VACÍO y correr los
fixtures del harness en cada tamaño. El relleno son los `title` que
Pydantic autogenera (`"title": "Encut"` para el campo `encut`) y que
`compact_json_schema` poda justamente para ahorrar bytes: no aportan
información al modelo, y para tamaños mayores se alargan con texto neutro.

Limitación honesta: ningún relleno es perfectamente inerte. Un title más
largo es ruido, y el ruido también puede degradar. Por eso el resultado se
lee como COTA: si a 6 KB la precisión aguanta, el techo real es >= 6 KB.
Si cae, no se puede afirmar que la culpa sea el tamaño y no el ruido.

Resultado de la corrida del 2026-08-05 (8 fixtures, 3 intentos, CPU pura),
que es la que fija el presupuesto vigente en `tests/test_router_parsing.py`:

    schema     qwen2.5:7b   gemma3:4b
    4122 B        8/8          7/8
    4884 B        8/8          7/8
    6000 B        8/8          7/8
    8004 B        8/8          7/8

Ni precisión ni latencia se mueven al duplicar el schema. `gemma3:4b`
—el modelo que el presupuesto viejo decía proteger— falla el mismo
fixture (`single_prepare_encut`) en TODOS los tamaños, incluido el actual:
su fallo no es de tamaño.

Uso:
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/calibrar_schema.py
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/calibrar_schema.py \
        qwen2.5:7b gemma3:4b
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from becario.infrastructure.ollama_router import (  # noqa: E402
    OllamaRouter, RouterDecision,
)
from scripts.live_router_check import (  # noqa: E402
    EditFixture, _check_edit, _check_route, load_fixtures,
)

_TARGETS = [None, 4900, 6000, 8000]  # None = el schema compacto de hoy
_ATTEMPTS = 3


def _con_titles(model) -> dict:
    """El schema SIN podar: Pydantic ya emite `title` en cada campo."""
    return model.model_json_schema()


def _inflar(schema: dict, objetivo: int) -> dict:
    """Alarga los `title` con texto neutro hasta pesar ~`objetivo` bytes."""
    s = copy.deepcopy(schema)
    props = s["$defs"]["RouterParams"]["properties"]
    relleno = 0
    while len(json.dumps(s)) < objetivo:
        creció = False
        for k, v in props.items():
            if len(json.dumps(s)) >= objetivo:
                break
            v["title"] = v.get("title", k) + " campo"
            creció = True
        relleno += 1
        if not creció or relleno > 200:
            break
    return s


def variantes() -> list[tuple[str, dict]]:
    compacto = OllamaRouter(base_url="x", model="x")._schema
    full = _con_titles(RouterDecision)
    out = [("compacto (hoy)", compacto), ("con titles", full)]
    for objetivo in _TARGETS[2:]:
        out.append((f"~{objetivo//1000}KB", _inflar(full, objetivo)))
    return out


def medir(modelo: str, url: str = "http://localhost:11434") -> None:
    fixtures = load_fixtures(REPO / "tests" / "fixtures" / "router")
    print(f"\n{'='*62}\nMODELO: {modelo}\n{'='*62}")
    router = OllamaRouter(base_url=url, model=modelo, timeout=300)
    try:
        router.route("hola")  # warm-up descartado
    except Exception:
        pass

    for nombre, schema in variantes():
        router._schema = schema
        size = len(json.dumps(schema))
        aciertos, lat, fallos = 0, [], []
        for fx in fixtures:
            checker = _check_edit if isinstance(fx, EditFixture) else _check_route
            ok = 0
            for _ in range(_ATTEMPTS):
                t0 = time.monotonic()
                try:
                    err = checker(router, fx)
                except Exception as exc:  # el schema puede romper la llamada
                    err = f"EXCEPCIÓN {type(exc).__name__}: {exc}"
                lat.append(time.monotonic() - t0)
                ok += err is None
            if ok > _ATTEMPTS // 2:
                aciertos += 1
            else:
                fallos.append(f"{fx.name}[{ok}/{_ATTEMPTS}]")
        marca = "✅" if aciertos == len(fixtures) else "⚠️ "
        print(f"{marca} {nombre:16} {size:5} B  ->  {aciertos}/{len(fixtures)}"
              f"  mediana {median(lat):5.1f}s")
        if fallos:
            print(f"    fallan: {', '.join(fallos)}")


def main() -> int:
    # Mismo cinturón de seguridad que `live_router_check.py`: pega contra un
    # Ollama REAL y no debe correr por accidente en pytest ni en CI.
    if os.environ.get("BECARIO_LIVE_ROUTER_CHECK") != "1":
        print(
            "Este script pega contra un Ollama REAL y hace ~100 llamadas por "
            "modelo.\nCorré con BECARIO_LIVE_ROUTER_CHECK=1 para confirmarlo."
        )
        return 1
    for modelo in sys.argv[1:] or ["qwen2.5:7b", "gemma3:4b"]:
        medir(modelo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
