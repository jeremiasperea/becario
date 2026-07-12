#!/usr/bin/env python3
"""Harness de validación en vivo del router de B.E.C.A.R.I.O. (tarea 6.1,
diseño §6.2, SR8: paridad de fixtures en `gemma3:4b` y `gemma4:12b`).

Pega contra un Ollama REAL en `localhost:11434` (o la URL que le pases) —
por eso corre deliberadamente FUERA de pytest/CI: `pyproject.toml` limita
`testpaths` a `tests/`, así que este archivo nunca se colecciona, y
además `main()` exige `BECARIO_LIVE_ROUTER_CHECK=1` como cinturón de
seguridad extra. Las funciones de parseo de fixtures (`parse_fixture` y
sus ayudantes) SÍ son puras y SÍ se testean en
`tests/test_live_router_fixtures.py`, sin red.

`gemma3:4b` es el peor caso (HC4 del diseño: banco de pruebas, modelo
más chico); `gemma4:12b` es el default de producción y no debe
regresar. Si un modelo falla un fixture, el harness lo reporta y termina
con exit code 1 — pensado para correrlo a mano antes de tocar el prompt
o el schema del router.

Uso:
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/live_router_check.py
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/live_router_check.py \
        --models gemma3:4b,gemma4:12b --url http://localhost:11434

Fixtures: `tests/fixtures/router/{single,multi,edit}_*.txt` — formato
en `parse_fixture()`.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.infrastructure.ollama_router import OllamaRouter  # noqa: E402

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "router"
# gemma3:4b primero: es el peor caso (HC4) y el que gatea el presupuesto
# de schema/prompt; gemma4:12b es el default de producción.
_DEFAULT_MODELS = ("gemma3:4b", "gemma4:12b")


@dataclass(frozen=True)
class RouteFixture:
    """Fixture de `router.route()`: un pedido (single o multi-paso) y la
    secuencia de `Intent` esperada, en orden."""

    name: str
    prompt: str
    expected_steps: list[str]


@dataclass(frozen=True)
class EditFixture:
    """Fixture de `router.extract_edit()`: un plan pendiente (contexto
    enumerado) + un mensaje de cambio, y a qué paso/params debería
    resolver — `expected_target_index=None` marca el caso ambiguo (SR6:
    "nunca se adivina")."""

    name: str
    plan_context: str
    prompt: str
    expected_target_index: Optional[int]
    expected_params: dict


Fixture = Union[RouteFixture, EditFixture]


def _parse_step_list(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _coerce_value(value: str):
    """Los params que extrae el router llegan tipados (Pydantic): los del
    fixture deben compararse con el mismo tipo, no como texto."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_params(raw: str) -> dict:
    pairs = [p.strip() for p in raw.split(",") if p.strip()]
    return {k: _coerce_value(v) for k, v in (p.split("=", 1) for p in pairs)}


def _parse_plan_context(raw: str) -> str:
    """`"stepA(k=v) | stepB(k=v)"` -> el mismo formato numerado que
    `BecarioService._render_plan_context` (application/services.py) le
    manda al LLM en producción — así el fixture ejercita el prompt real,
    no una aproximación."""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return "\n".join(f"{i}. {p}" for i, p in enumerate(parts, start=1))


def _kv_lines(text: str) -> dict[str, str]:
    """Parseo genérico línea a línea `clave: valor` (ignora comentarios
    `#` y líneas vacías). Sin valores multilínea a propósito: los
    fixtures son chicos y legibles, uno por caso."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def parse_fixture(text: str, name: str) -> Fixture:
    """`kind: edit` produce un `EditFixture`; cualquier otro valor (o su
    ausencia) produce un `RouteFixture` — así los fixtures `single_*`/
    `multi_*` no necesitan declarar `kind` explícitamente."""
    fields = _kv_lines(text)
    if fields.get("kind") == "edit":
        raw_target = fields.get("target_index", "").strip()
        target_index = None if raw_target in ("", "null") else int(raw_target)
        return EditFixture(
            name=name,
            plan_context=_parse_plan_context(fields.get("plan", "")),
            prompt=fields.get("prompt", ""),
            expected_target_index=target_index,
            expected_params=_parse_params(fields.get("params", "")),
        )
    return RouteFixture(
        name=name,
        prompt=fields.get("prompt", ""),
        expected_steps=_parse_step_list(fields.get("steps", "")),
    )


def load_fixtures(fixtures_dir: Path = _FIXTURES_DIR) -> list[Fixture]:
    return [
        parse_fixture(path.read_text(encoding="utf-8"), name=path.name)
        for path in sorted(fixtures_dir.glob("*.txt"))
    ]


def _check_route(router: OllamaRouter, fx: RouteFixture) -> Optional[str]:
    plan = router.route(fx.prompt)
    actual = [s.action.value for s in plan.steps]
    if actual != fx.expected_steps:
        return f"esperaba steps={fx.expected_steps}, obtuve {actual}"
    return None


def _check_edit(router: OllamaRouter, fx: EditFixture) -> Optional[str]:
    target_index, params = router.extract_edit(fx.plan_context, fx.prompt)
    if target_index != fx.expected_target_index:
        return f"esperaba target_index={fx.expected_target_index}, obtuve {target_index}"
    if fx.expected_target_index is not None and params != fx.expected_params:
        return f"esperaba params={fx.expected_params}, obtuve {params}"
    return None


def _installed_models(url: str) -> Optional[set[str]]:
    """Nombres de modelos disponibles en el Ollama destino, o None si no
    se pudo consultar (en ese caso no se saltea nada: mejor intentar)."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=10) as resp:
            data = json.load(resp)
        return {m.get("name", "") for m in data.get("models", [])}
    except OSError:
        return None


def run(models: list[str], url: str, fixtures_dir: Path = _FIXTURES_DIR) -> bool:
    """Corre TODOS los fixtures contra CADA modelo; imprime un reporte
    legible y devuelve `True` solo si todo pasó en todos los modelos."""
    fixtures = load_fixtures(fixtures_dir)
    if not fixtures:
        print(f"⚠️  No hay fixtures en {fixtures_dir}")
        return False
    installed = _installed_models(url)
    all_ok = True
    for model in models:
        if installed is not None and model not in installed:
            # Un modelo ausente no es un fallo de clasificación: se salta
            # con aviso para no reportar ❌ engañosos (p. ej. gemma4:12b
            # solo existe en la máquina de producción).
            print(f"\n== modelo: {model} ==")
            print(f"⏭️  no instalado en {url} — salteado (parcial, no falla)")
            continue
        router = OllamaRouter(base_url=url, model=model)
        print(f"\n== modelo: {model} ==")
        for fx in fixtures:
            checker = _check_edit if isinstance(fx, EditFixture) else _check_route
            error = checker(router, fx)
            status = "✅" if error is None else "❌"
            print(f"{status} {fx.name}" + (f" — {error}" if error else ""))
            all_ok = all_ok and error is None
    return all_ok


def main() -> int:
    if os.environ.get("BECARIO_LIVE_ROUTER_CHECK") != "1":
        print(
            "Este harness pega contra un Ollama REAL — no corre en CI ni "
            "en el pytest por defecto.\nCorré con "
            "BECARIO_LIVE_ROUTER_CHECK=1 para confirmarlo explícitamente."
        )
        return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(_DEFAULT_MODELS))
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ok = run(models, args.url)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
