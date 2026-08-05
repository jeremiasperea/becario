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

Cada fixture se evalúa por MAYORÍA sobre `--attempts` intentos (default
3, impar). Motivo (AR-2 / ADR-0006): `gemma4:12b` sobre CPU es
no-determinístico — el greedy decoding con `temperature=0` NO es
bit-reproducible entre threads (el orden de reducción de floats varía),
así que el 12b flakea ~1 de cada 6 en composición multi-paso aun con el
mismo input. Un solo intento daría ❌ engañosos; la mayoría estabiliza el
gate manual. El voto se imprime entre corchetes cuando no fue unánime
(p. ej. `✅ multi_destructive_tail [2/3]`).

Además de acertar, cada fixture reporta su latencia (mediana de los
intentos) y cada modelo su mediana por llamada. En CPU la velocidad es
parte de la decisión de qué modelo servir, no un detalle: un modelo que
acierta pero tarda 40s por mensaje no es usable desde Telegram. Antes de
medir se hace una generación de warm-up descartada, porque la primera
llamada paga la carga del modelo a RAM.

Con `--json` el resultado además se vuelca como scoreboard versionable
(`build_scoreboard()`): puntaje por modelo, latencia mediana y detalle por
fixture. Ese archivo es el que se commitea y compara entre corridas — la
consola sola no deja comparar 4/6 contra 6/6 sin que alguien lo anote a mano.

Uso:
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/live_router_check.py
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/live_router_check.py \
        --models gemma3:4b,gemma4:12b --url http://localhost:11434 --attempts 5
    BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/live_router_check.py \
        --models qwen2.5-coder:14b --timeout 300 --json docs/scoreboard_router.json

Fixtures: `tests/fixtures/router/{single,multi,edit}_*.txt` — formato
en `parse_fixture()`. Con `--fixtures-dir` se puede apuntar a otro set, por
ejemplo el que exporta `scripts/export_router_dataset.py --fixtures` con las
decisiones confirmadas por humanos.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.infrastructure.ollama_router import OllamaRouter  # noqa: E402

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "router"
# gemma3:4b primero: es el peor caso (HC4) y el que gatea el presupuesto
# de schema/prompt; gemma4:12b sigue siendo el default de `config.py` y no
# debe regresar, aunque esté medido en 4/6 a 77.9s.
# qwen2.5:7b es la línea base rápida (6/6 a 5.4s) y qwen2.5-coder:14b es el
# que sirve producción hoy: ganó el benchmark de 13 casos del 2026-08-01 por
# inventar menos. gemma4:e4b salió de la lista (6/6 pero 27s, superado por
# los dos qwen).
_DEFAULT_MODELS = ("gemma3:4b", "qwen2.5:7b", "qwen2.5-coder:14b", "gemma4:12b")


@dataclass(frozen=True)
class RouteFixture:
    """Fixture de `router.route()`: un pedido (single o multi-paso) y la
    secuencia de `Intent` esperada, en orden.

    `expected_params` (opcional, solo fixtures de UN paso) exige que esos
    pares estén presentes en los parámetros extraídos (chequeo de
    subconjunto: extras como tipo_calculo no fallan). Sin esto, un
    fixture de preparar_calculo pasaba en verde aunque el modelo
    perdiera formula/red_cristalina — el bug real que motivó agregarlo."""

    name: str
    prompt: str
    expected_steps: list[str]
    expected_params: dict = field(default_factory=dict)


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


@dataclass(frozen=True)
class FixtureResult:
    """Lo que un fixture dio contra UN modelo: si pasó la mayoría, con qué
    votos, cuánto tardó cada intento y el error más frecuente si falló.

    Guarda las latencias crudas (no solo su mediana) porque la mediana del
    modelo se calcula sobre los intentos individuales, no sobre las medianas
    por fixture: promediar medianas le daría el mismo peso a un fixture de 3
    intentos que a uno de 5."""

    name: str
    passes: int
    attempts: int
    latencies: tuple[float, ...]
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def latency_seconds(self) -> float:
        return median(self.latencies) if self.latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "passes": self.passes,
            "attempts": self.attempts,
            "latency_seconds": round(self.latency_seconds, 3),
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelScore:
    """El puntaje de un modelo sobre el set completo de fixtures.

    `skipped=True` marca un modelo que no está instalado en el Ollama
    destino: no se lo evalúa y NO cuenta como fallo (mismo criterio que la
    consola, que imprime ⏭️). Un 0/0 salteado y un 0/6 real son cosas
    distintas y el scoreboard tiene que poder distinguirlas."""

    model: str
    results: tuple[FixtureResult, ...] = ()
    skipped: bool = False

    @property
    def hits(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> bool:
        return self.skipped or all(r.ok for r in self.results)

    @property
    def median_latency_seconds(self) -> Optional[float]:
        latencies = [lat for r in self.results for lat in r.latencies]
        return median(latencies) if latencies else None

    def to_dict(self) -> dict:
        median_latency = self.median_latency_seconds
        return {
            "model": self.model,
            "skipped": self.skipped,
            "hits": self.hits,
            "total": self.total,
            "median_latency_seconds": (
                None if median_latency is None else round(median_latency, 3)
            ),
            "fixtures": [r.to_dict() for r in self.results],
        }


def build_scoreboard(
    scores: list[ModelScore], *, attempts: int, fixtures_dir: Path
) -> dict:
    """El scoreboard versionable: qué se midió, contra qué y con qué
    resultado. `fixtures_dir` va relativo a la raíz del repo para que el
    archivo no filtre la ruta absoluta de la máquina que lo generó."""
    root = Path(__file__).resolve().parent.parent
    try:
        where = fixtures_dir.resolve().relative_to(root).as_posix()
    except ValueError:
        where = fixtures_dir.as_posix()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempts": attempts,
        "fixtures_dir": where,
        "fixtures": sorted({r.name for s in scores for r in s.results}),
        "models": [s.to_dict() for s in scores],
    }


def check_scoreboard(board, fixture_names: set[str]) -> list[str]:
    """Los problemas de un scoreboard commiteado; lista vacía = está al día.

    Es el gate BARATO que sí puede correr en CI: valida forma y frescura sin
    red y sin modelo. El gate caro —correr los fixtures contra un LLM— es
    local por costo: con `gemma4:12b` medido en 77.9s por llamada, 30 fixtures
    por 3 intentos son ~117 minutos, y los runners no tienen ni GPU ni el
    modelo instalado.

    La comprobación de frescura es la que atrapa el olvido real: si alguien
    agrega un fixture y no vuelve a correr el harness, el scoreboard deja de
    describir lo que hay en `tests/fixtures/router/` y CI lo dice."""
    problems: list[str] = []
    if not isinstance(board, dict):
        return ["el scoreboard no es un objeto JSON"]
    for key in ("generated_at", "attempts", "fixtures_dir", "fixtures", "models"):
        if key not in board:
            problems.append(f"falta la clave '{key}'")
    if problems:
        return problems

    if not isinstance(board["attempts"], int) or board["attempts"] < 1:
        problems.append(f"'attempts' inválido: {board['attempts']!r}")

    listed = board["fixtures"]
    if not isinstance(listed, list):
        problems.append("'fixtures' no es una lista")
    else:
        faltan = sorted(fixture_names - set(listed))
        sobran = sorted(set(listed) - fixture_names)
        if faltan:
            problems.append(
                f"fixtures sin medir (corré el harness con --json): {faltan}"
            )
        if sobran:
            problems.append(f"fixtures medidos que ya no existen: {sobran}")

    models = board["models"]
    if not isinstance(models, list) or not models:
        problems.append("'models' vacío: el scoreboard no midió ningún modelo")
        return problems

    for entry in models:
        name = entry.get("model", "?") if isinstance(entry, dict) else "?"
        if not isinstance(entry, dict):
            problems.append(f"entrada de modelo inválida: {entry!r}")
            continue
        for key in ("model", "skipped", "hits", "total", "fixtures"):
            if key not in entry:
                problems.append(f"{name}: falta la clave '{key}'")
        if "hits" in entry and "total" in entry and entry["hits"] > entry["total"]:
            problems.append(f"{name}: hits ({entry['hits']}) > total ({entry['total']})")

    if all(entry.get("skipped") for entry in models if isinstance(entry, dict)):
        problems.append(
            "todos los modelos salteados: el scoreboard no mide nada "
            "(¿estaba Ollama abajo?)"
        )
    return problems


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
        expected_params=_parse_params(fields.get("params", "")),
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
    if fx.expected_params:
        got = plan.steps[0].parametros
        missing = {
            k: v for k, v in fx.expected_params.items() if got.get(k) != v
        }
        if missing:
            return f"faltan params {missing}, obtuve {got}"
    return None


def _check_edit(router: OllamaRouter, fx: EditFixture) -> Optional[str]:
    target_index, params = router.extract_edit(fx.plan_context, fx.prompt)
    if target_index != fx.expected_target_index:
        return f"esperaba target_index={fx.expected_target_index}, obtuve {target_index}"
    if fx.expected_target_index is not None and params != fx.expected_params:
        return f"esperaba params={fx.expected_params}, obtuve {params}"
    return None


def _majority_check(
    checker: Callable[[OllamaRouter, Fixture], Optional[str]],
    router: OllamaRouter,
    fx: Fixture,
    attempts: int,
) -> tuple[Optional[str], int, list[float]]:
    """Corre `checker` `attempts` veces y decide por MAYORÍA. Devuelve
    `(error, passes, latencias)`: `error=None` si la mayoría pasó, y una
    latencia en segundos por intento. Necesario porque `gemma4:12b` sobre
    CPU es no-determinístico (ver docstring del módulo); un intento único
    daría ❌ engañosos. Ante mayoría fallida, reporta el mensaje de error
    más frecuente.

    Las latencias incluyen los intentos fallidos a propósito: un modelo
    que responde rápido pero mal no es más barato, y esconder ese costo
    falsearía la comparación."""
    passes = 0
    errors: list[str] = []
    latencies: list[float] = []
    for _ in range(attempts):
        started = time.monotonic()
        err = checker(router, fx)
        latencies.append(time.monotonic() - started)
        if err is None:
            passes += 1
        else:
            errors.append(err)
    if passes > attempts // 2:
        return None, passes, latencies
    return Counter(errors).most_common(1)[0][0], passes, latencies


def _installed_models(url: str) -> Optional[set[str]]:
    """Nombres de modelos disponibles en el Ollama destino, o None si no
    se pudo consultar (en ese caso no se saltea nada: mejor intentar)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=10) as resp:
            data = json.load(resp)
        return {m.get("name", "") for m in data.get("models", [])}
    except OSError:
        return None


def run(
    models: list[str],
    url: str,
    fixtures_dir: Path = _FIXTURES_DIR,
    attempts: int = 3,
    timeout: float = 120.0,
) -> list[ModelScore]:
    """Corre TODOS los fixtures contra CADA modelo; imprime el mismo reporte
    legible de siempre y devuelve el puntaje por modelo.

    Devuelve una LISTA y no un `bool` para que el resultado se pueda volcar a
    JSON y compararse entre corridas: el semáforo pasa/falla no distingue 4/6
    de 6/6, que es justo la diferencia que decide qué modelo servir. El exit
    code lo deriva `main()` de estos puntajes.

    Una lista vacía significa que no se midió NADA (sin fixtures o sin
    modelos) y `main()` la trata como fallo: "no medí" no es "está todo bien".

    Cada fixture se decide por mayoría sobre `attempts` intentos. `timeout` es
    el tope por request al LLM (subilo en CPU: un 12b puede superar los 120s
    por generación y dar timeouts espurios)."""
    fixtures = load_fixtures(fixtures_dir)
    if not fixtures:
        print(f"⚠️  No hay fixtures en {fixtures_dir}")
        return []
    installed = _installed_models(url)
    scores: list[ModelScore] = []
    for model in models:
        print(f"\n== modelo: {model} ==")
        if installed is not None and model not in installed:
            # Un modelo ausente no es un fallo de clasificación: se salta
            # con aviso para no reportar ❌ engañosos (p. ej. gemma4:12b
            # solo existe en la máquina de producción).
            print(f"⏭️  no instalado en {url} — salteado (parcial, no falla)")
            scores.append(ModelScore(model=model, skipped=True))
            continue
        router = OllamaRouter(base_url=url, model=model, timeout=timeout)
        # Warm-up descartado: la primera generación paga la carga del
        # modelo a RAM, que en CPU domina el reloj y crece con el tamaño.
        # Sin esto, el primer fixture sale inflado y la comparación entre
        # modelos castiga dos veces al más grande.
        try:
            router.route("hola")
        except Exception:
            # El warm-up no decide nada: si falla, los fixtures lo dirán.
            pass
        results: list[FixtureResult] = []
        for fx in fixtures:
            checker = _check_edit if isinstance(fx, EditFixture) else _check_route
            error, passes, latencies = _majority_check(checker, router, fx, attempts)
            result = FixtureResult(
                name=fx.name,
                passes=passes,
                attempts=attempts,
                latencies=tuple(latencies),
                error=error,
            )
            results.append(result)
            status = "✅" if result.ok else "❌"
            # El voto solo se muestra si no fue unánime: hace visible el
            # flake (p. ej. `✅ multi_destructive_tail [2/3]`) sin ruido
            # cuando todo coincidió.
            vote = "" if passes == attempts else f" [{passes}/{attempts}]"
            # Mediana y no promedio: con `attempts` chico un solo outlier
            # (scheduler, swap) arrastraría el promedio y haría ruido.
            secs = f" {result.latency_seconds:5.1f}s"
            print(f"{status} {fx.name}{vote}{secs}" + (f" — {error}" if error else ""))
        score = ModelScore(model=model, results=tuple(results))
        scores.append(score)
        calls = [lat for r in results for lat in r.latencies]
        if calls:
            print(
                f"   ── {score.hits}/{score.total}"
                f" · mediana {median(calls):.1f}s/llamada"
                f" · {sum(calls):.0f}s en {len(calls)} llamadas"
            )
    return scores


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
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="intentos por fixture; decide por mayoría (impar recomendado)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="tope por request al LLM en segundos (subilo en CPU para 12b)",
    )
    parser.add_argument(
        "--fixtures-dir",
        metavar="DIR",
        default=str(_FIXTURES_DIR),
        help=(
            "directorio de fixtures a evaluar (default: tests/fixtures/router). "
            "Apuntalo al DIR que escribió export_router_dataset.py --fixtures "
            "para correr los casos reales confirmados por humanos."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="además del reporte en consola, escribir el scoreboard en JSON",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts debe ser >= 1")
    if args.timeout <= 0:
        parser.error("--timeout debe ser > 0")
    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.is_dir():
        parser.error(f"--fixtures-dir no es un directorio: {fixtures_dir}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    scores = run(
        models,
        args.url,
        fixtures_dir=fixtures_dir,
        attempts=args.attempts,
        timeout=args.timeout,
    )
    if args.json:
        scoreboard = build_scoreboard(
            scores, attempts=args.attempts, fixtures_dir=fixtures_dir
        )
        # `\n` final para que el archivo sea POSIX y no ensucie el diff cuando
        # se commitea el scoreboard.
        Path(args.json).write_text(
            json.dumps(scoreboard, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nScoreboard escrito en {args.json}")
    # Sin puntajes no se midió nada: eso es un fallo del gate, no un pase.
    ok = bool(scores) and all(s.ok for s in scores)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
