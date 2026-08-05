#!/usr/bin/env python3
"""Exporta el registro de decisiones del router como dataset de evaluación.

Lee la tabla `decisiones_router` (ver `SQLiteRouterDecisionLog`) y emite:

- **JSONL** (default): una decisión por línea, con texto, pasos, modelo,
  latencia y desenlace. Es el formato de trabajo para curar el set de
  evaluación (revisar los 'routed' a mano, descartar ruido) y, a futuro,
  la materia prima de un fine-tuning.
- **Fixtures** (`--fixtures DIR`): las decisiones CONFIRMADAS como archivos
  `prompt:`/`steps:`/`params:` compatibles con `scripts/live_router_check.py`
  — cada plan que un humano avaló se vuelve un caso de regresión del router.
  Los planes de un solo paso llevan `params:` con los campos representables:
  sin esa línea el fixture pasa en verde aunque el modelo pierda `formula`,
  que es exactamente el fallo que se quiere medir. Correr el harness sobre
  ellos con `live_router_check.py --fixtures-dir DIR`.

Solo las confirmadas se exportan como fixtures: son las únicas con
etiqueta humana positiva. 'routed' sin desenlace no prueba nada (el
usuario pudo ignorar una respuesta absurda) y 'cancelled'/'error'
necesitan revisión manual antes de servir de ground truth.

Uso:
    .venv/bin/python scripts/export_router_dataset.py                # JSONL a stdout
    .venv/bin/python scripts/export_router_dataset.py --db becario.db --outcome confirmed
    .venv/bin/python scripts/export_router_dataset.py --fixtures tests/fixtures/router_real/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.infrastructure.storage import SQLiteRouterDecisionLog  # noqa: E402

_OUTCOMES = ("routed", "confirmed", "cancelled", "error")


def _slug(text: str, max_len: int = 40) -> str:
    """Nombre de archivo legible a partir del texto del mensaje."""
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", norm).strip("_").lower()
    return norm[:max_len] or "decision"


def _is_representable(value) -> bool:
    """¿El formato `k=v, k=v` de los fixtures puede llevar este valor sin
    ambigüedad, y `_coerce_value()` del harness lo devuelve igual?

    Se descartan a propósito:

    - **Listas y diccionarios** (`puntos_k=[1,1,1]`, `miller`, `tags_incar`):
      `_parse_params()` corta por `,` y por `=`, así que un valor con comas
      adentro se parsearía como varios pares y produciría un fixture roto.
    - **Booleanos**: viajarían como `True` y el harness los leería como el
      texto `"True"`, fallando la comparación contra el `bool` real.
    - **Strings que parecen números** (`nombre_trabajo="123"`): `_coerce_value`
      los convertiría a `int` y la comparación contra el `str` original daría
      un ❌ falso.

    Perder esos campos NO deja el fixture ciego a lo que importa: `formula` y
    `red_cristalina` —los que los modelos locales sueltan (0/3 medido)— son
    strings simples y sí se representan."""
    if isinstance(value, bool) or value in (None, "", [], {}):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    if any(sep in value for sep in (",", "=", "\n")):
        return False
    return _coerce_value(value) == value


def _coerce_value(value: str):
    """Espejo de `_coerce_value()` en `scripts/live_router_check.py`: se
    duplica en vez de importarse para que el exportador no dependa del
    harness (son dos scripts sueltos, ninguno importa al otro)."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def format_params(parametros: dict) -> str:
    """Los parámetros de un paso como la línea `params:` de un fixture, en el
    formato que `_parse_params()` del harness ya sabe leer. Devuelve `""` si
    ningún campo es representable."""
    pairs = [
        f"{k}={v}" for k, v in sorted(parametros.items()) if _is_representable(v)
    ]
    return ", ".join(pairs)


def export_jsonl(rows: list[dict], out) -> int:
    for row in rows:
        record = {
            "text": row["text"],
            "steps": json.loads(row["steps_json"]),
            "outcome": row["outcome"],
            "model": row["model"],
            "latency_seconds": row["latency_seconds"],
            "created_at": row["created_at"],
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(rows)


def render_fixture(row: dict) -> Optional[str]:
    """Una decisión confirmada como fixture del harness, o `None` si no se
    puede representar.

    Los planes de UN paso llevan además `params:`. Sin esa línea el fixture es
    ciego justo al fallo que motiva medir: el docstring de `RouteFixture` en
    `scripts/live_router_check.py` lo dice —"un fixture de preparar_calculo
    pasaba en verde aunque el modelo perdiera formula/red_cristalina"—. Los
    planes multi-paso siguen saliendo solo con `steps:` porque
    `expected_params` solo mira `steps[0]`: exportar los params de un plan
    compuesto prometería una cobertura que el chequeo no da.

    Un mensaje con salto de línea se descarta: los fixtures se parsean línea a
    línea (`_kv_lines`), así que ese archivo saldría corrupto y en silencio."""
    text = row["text"]
    if "\n" in text or not text.strip():
        return None
    steps = json.loads(row["steps_json"])
    if not steps:
        return None
    actions = ", ".join(s["action"] for s in steps)
    body = (
        f"# Confirmado por un humano el {row['created_at']} "
        f"(modelo {row['model']}, decisión #{row['id']}).\n"
        f"prompt: {text}\n"
        f"steps: {actions}\n"
    )
    if len(steps) == 1:
        params = format_params(steps[0].get("parametros") or {})
        if params:
            body += f"params: {params}\n"
    return body


def export_fixtures(rows: list[dict], out_dir: Path) -> tuple[int, int]:
    """Decisiones confirmadas → fixtures del harness. Devuelve
    `(escritos, descartados)`; los descartados son los que `render_fixture`
    no pudo representar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for row in rows:
        if row["outcome"] != "confirmed":
            continue
        body = render_fixture(row)
        if body is None:
            skipped += 1
            continue
        path = out_dir / f"real_{row['id']:05d}_{_slug(row['text'])}.txt"
        path.write_text(body, encoding="utf-8")
        written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="becario.db", help="ruta de la base SQLite")
    parser.add_argument(
        "--outcome", choices=_OUTCOMES, default=None,
        help="exportar solo este desenlace (default: todos)",
    )
    parser.add_argument(
        "--fixtures", metavar="DIR", default=None,
        help="además del JSONL, escribir las CONFIRMADAS como fixtures del harness",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"No existe la base {args.db!r}", file=sys.stderr)
        return 1

    # El modelo del constructor es solo para escribir; acá se lee.
    log = SQLiteRouterDecisionLog(args.db, model="")
    rows = log.rows(outcome=args.outcome)
    count = export_jsonl(rows, sys.stdout)
    print(f"{count} decisiones exportadas", file=sys.stderr)

    if args.fixtures:
        confirmed = [r for r in rows if r["outcome"] == "confirmed"]
        written, skipped = export_fixtures(confirmed, Path(args.fixtures))
        print(f"{written} fixtures confirmados en {args.fixtures}", file=sys.stderr)
        if skipped:
            print(
                f"{skipped} descartados: mensaje multilínea o plan vacío "
                "(no representables en el formato de fixture)",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
