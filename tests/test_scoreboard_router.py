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

from scripts.live_router_check import (
    check_scoreboard,
    load_fixtures,
    router_fingerprint,
)

_ROOT = Path(__file__).resolve().parent.parent
_SCOREBOARD = _ROOT / "docs" / "scoreboard_router.json"
_FIXTURES_DIR = _ROOT / "tests" / "fixtures" / "router"

# Cuánto puede envejecer una medición antes de dejar de valer como
# evidencia. La huella (`router_fingerprint`) cubre los cambios de código,
# pero un modelo puede cambiar de comportamiento sin que se mueva una línea
# —pasó: entre el 2026-08-04 y el 08-05 un fixture pasó de fallar de una
# forma a fallar de otra— y contra eso solo sirve volver a medir. 30 días
# es el compromiso: lo bastante largo para no molestar en cada PR, lo
# bastante corto para que una deriva no viva meses en silencio. Re-correr
# cuesta ~11 minutos con los dos modelos.
_MAX_SCOREBOARD_AGE_DAYS = 30


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


class TestTheContractTheBoardWasMeasuredAgainst:
    """La huella cierra el agujero por el que se coló la mutación: el
    tablero seguía pareciendo válido mientras describía un router al que ya
    le habían cambiado el schema."""

    def test_a_board_measured_with_another_contract_is_caught(self):
        board = _board(router_fingerprint="viejo123")
        problems = check_scoreboard(
            board, {"a.txt", "b.txt"}, fingerprint="nuevo456"
        )
        assert any("otro contrato de router" in p for p in problems)

    def test_a_matching_fingerprint_passes(self):
        board = _board(router_fingerprint="igual789")
        assert check_scoreboard(
            board, {"a.txt", "b.txt"}, fingerprint="igual789"
        ) == []

    def test_a_board_from_before_the_fingerprint_existed_is_caught(self):
        problems = check_scoreboard(_board(), {"a.txt", "b.txt"}, fingerprint="x")
        assert any("con qué contrato" in p for p in problems)

    def test_the_fingerprint_moves_when_the_router_prompt_moves(self, monkeypatch):
        from becario.infrastructure import ollama_router as router_mod

        antes = router_fingerprint(_FIXTURES_DIR)
        monkeypatch.setattr(
            router_mod, "_SYSTEM_PROMPT", router_mod._SYSTEM_PROMPT + "\nregla nueva"
        )
        assert router_fingerprint(_FIXTURES_DIR) != antes

    def test_the_fingerprint_moves_when_a_fixture_is_edited(self, tmp_path):
        (tmp_path / "uno.txt").write_text("prompt: a\nsteps: consultar_db\n")
        antes = router_fingerprint(tmp_path)
        (tmp_path / "uno.txt").write_text("prompt: OTRA COSA\nsteps: consultar_db\n")
        assert router_fingerprint(tmp_path) != antes

    def test_the_fingerprint_is_stable_across_calls(self):
        assert router_fingerprint(_FIXTURES_DIR) == router_fingerprint(_FIXTURES_DIR)


class TestTheBoardGoesStaleOnItsOwn:
    """Un modelo puede cambiar de comportamiento sin que se mueva una línea
    de código. Contra eso no hay huella que sirva: solo volver a medir."""

    def test_an_old_board_is_caught(self):
        board = _board(generated_at="2020-01-01T00:00:00+00:00")
        problems = check_scoreboard(board, {"a.txt", "b.txt"}, max_age_days=30)
        assert any("días" in p for p in problems)

    def test_a_recent_board_passes(self):
        from datetime import datetime, timezone

        ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
        board = _board(generated_at=ahora)
        assert check_scoreboard(board, {"a.txt", "b.txt"}, max_age_days=30) == []

    def test_an_unreadable_date_is_caught(self):
        board = _board(generated_at="ayer a la tarde")
        problems = check_scoreboard(board, {"a.txt", "b.txt"}, max_age_days=30)
        assert any("ilegible" in p for p in problems)


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

    def test_describes_the_router_that_exists_today(self):
        # El gate completo: fixtures, contrato del router y antigüedad. Si
        # falla, la respuesta es SIEMPRE la misma —volver a medir— y el
        # mensaje del problema dice cuál de las tres cosas se venció.
        board = json.loads(_SCOREBOARD.read_text(encoding="utf-8"))
        names = {fx.name for fx in load_fixtures(_FIXTURES_DIR)}
        problems = check_scoreboard(
            board,
            names,
            fingerprint=router_fingerprint(_FIXTURES_DIR),
            max_age_days=_MAX_SCOREBOARD_AGE_DAYS,
        )
        assert problems == [], (
            "el tablero del router está vencido:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRe-corré (~11 min, necesita Ollama arriba):\n"
            "  BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python "
            "scripts/live_router_check.py --json docs/scoreboard_router.json"
        )
