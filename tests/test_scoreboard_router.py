"""Gate barato del scoreboard del router: valida forma y frescura del
archivo commiteado, sin red y sin modelo.

El gate CARO —correr los fixtures contra un LLM real— es local a propósito:
con `gemma4:12b` medido en 77.9s por llamada (`docs/comparacion_modelos.md`),
30 fixtures por 3 intentos son ~117 minutos, y los runners de CI no tienen ni
GPU ni los modelos instalados. Lo que CI sí puede hacer es exigir que el
resultado de esa corrida local esté commiteado y describa los fixtures que hay
hoy en `tests/fixtures/router/`."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.live_router_check import check_scoreboard, load_fixtures

_ROOT = Path(__file__).resolve().parent.parent
_SCOREBOARD = _ROOT / "docs" / "scoreboard_router.json"
_FIXTURES_DIR = _ROOT / "tests" / "fixtures" / "router"


def _board(**overrides) -> dict:
    board = {
        "generated_at": "2026-08-04T12:00:00+00:00",
        "attempts": 3,
        "fixtures_dir": "tests/fixtures/router",
        "fixtures": ["a.txt", "b.txt"],
        "models": [
            {
                "model": "qwen2.5-coder:14b",
                "skipped": False,
                "hits": 2,
                "total": 2,
                "median_latency_seconds": 4.2,
                "fixtures": [],
            }
        ],
    }
    board.update(overrides)
    return board


class TestCheckScoreboard:
    def test_a_well_formed_scoreboard_has_no_problems(self):
        assert check_scoreboard(_board(), {"a.txt", "b.txt"}) == []

    def test_a_new_fixture_nobody_measured_is_caught(self):
        # El olvido real: se agrega un fixture y no se vuelve a correr el
        # harness. El scoreboard queda describiendo otra cosa.
        problems = check_scoreboard(_board(), {"a.txt", "b.txt", "nuevo.txt"})
        assert len(problems) == 1
        assert "nuevo.txt" in problems[0]

    def test_a_deleted_fixture_still_in_the_scoreboard_is_caught(self):
        problems = check_scoreboard(_board(), {"a.txt"})
        assert any("b.txt" in p for p in problems)

    def test_missing_keys_are_reported(self):
        board = _board()
        del board["models"]
        assert check_scoreboard(board, set()) == ["falta la clave 'models'"]

    def test_not_an_object_is_reported(self):
        assert check_scoreboard([], set()) == ["el scoreboard no es un objeto JSON"]

    def test_invalid_attempts_is_reported(self):
        problems = check_scoreboard(_board(attempts=0), {"a.txt", "b.txt"})
        assert any("attempts" in p for p in problems)

    def test_no_models_is_reported(self):
        problems = check_scoreboard(_board(models=[]), {"a.txt", "b.txt"})
        assert any("no midió ningún modelo" in p for p in problems)

    def test_hits_above_total_is_reported(self):
        board = _board()
        board["models"][0]["hits"] = 5
        problems = check_scoreboard(board, {"a.txt", "b.txt"})
        assert any("hits" in p for p in problems)

    def test_model_entry_missing_a_key_is_reported(self):
        board = _board()
        del board["models"][0]["hits"]
        problems = check_scoreboard(board, {"a.txt", "b.txt"})
        assert any("falta la clave 'hits'" in p for p in problems)

    def test_all_models_skipped_measures_nothing(self):
        # Correr el harness con Ollama abajo produce un scoreboard entero de
        # ⏭️: parece verde y no midió nada.
        board = _board(
            models=[
                {
                    "model": "qwen2.5-coder:14b", "skipped": True, "hits": 0,
                    "total": 0, "median_latency_seconds": None, "fixtures": [],
                }
            ]
        )
        problems = check_scoreboard(board, {"a.txt", "b.txt"})
        assert any("no mide nada" in p for p in problems)


@pytest.mark.skipif(
    not _SCOREBOARD.exists(),
    reason=(
        f"todavía no hay {_SCOREBOARD.relative_to(_ROOT)}: generalo con "
        "BECARIO_LIVE_ROUTER_CHECK=1 python scripts/live_router_check.py "
        "--json docs/scoreboard_router.json (necesita Ollama arriba)"
    ),
)
class TestCommittedScoreboard:
    def test_is_valid_json(self):
        json.loads(_SCOREBOARD.read_text(encoding="utf-8"))

    def test_describes_the_fixtures_that_exist_today(self):
        board = json.loads(_SCOREBOARD.read_text(encoding="utf-8"))
        names = {fx.name for fx in load_fixtures(_FIXTURES_DIR)}
        assert check_scoreboard(board, names) == []
