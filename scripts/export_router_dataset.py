#!/usr/bin/env python3
"""Exporta el registro de decisiones del router como dataset de evaluación.

Lee la tabla `decisiones_router` (ver `SQLiteRouterDecisionLog`) y emite:

- **JSONL** (default): una decisión por línea, con texto, pasos, modelo,
  latencia y desenlace. Es el formato de trabajo para curar el set de
  evaluación (revisar los 'routed' a mano, descartar ruido) y, a futuro,
  la materia prima de un fine-tuning.
- **Fixtures** (`--fixtures DIR`): las decisiones CONFIRMADAS como archivos
  `prompt:`/`steps:` compatibles con `scripts/live_router_check.py` — cada
  plan que un humano avaló se vuelve un caso de regresión del router.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.infrastructure.storage import SQLiteRouterDecisionLog  # noqa: E402

_OUTCOMES = ("routed", "confirmed", "cancelled", "error")


def _slug(text: str, max_len: int = 40) -> str:
    """Nombre de archivo legible a partir del texto del mensaje."""
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", norm).strip("_").lower()
    return norm[:max_len] or "decision"


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


def export_fixtures(rows: list[dict], out_dir: Path) -> int:
    """Decisiones confirmadas → fixtures `prompt:`/`steps:` del harness."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        if row["outcome"] != "confirmed":
            continue
        steps = json.loads(row["steps_json"])
        actions = ", ".join(s["action"] for s in steps)
        body = (
            f"# Confirmado por un humano el {row['created_at']} "
            f"(modelo {row['model']}, decisión #{row['id']}).\n"
            f"prompt: {row['text']}\n"
            f"steps: {actions}\n"
        )
        path = out_dir / f"real_{row['id']:05d}_{_slug(row['text'])}.txt"
        path.write_text(body, encoding="utf-8")
        written += 1
    return written


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
        written = export_fixtures(confirmed, Path(args.fixtures))
        print(f"{written} fixtures confirmados en {args.fixtures}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
