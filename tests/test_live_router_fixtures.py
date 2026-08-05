"""Tests de las funciones PURAS del harness de validación en vivo del
router (tarea 6.1, SR8). Estos tests corren siempre, sin red: validan el
parseo de los fixtures en `tests/fixtures/router/`. El harness en sí
(`scripts/live_router_check.py` invocado como script) SÍ pega contra un
Ollama real y está deliberadamente fuera del pytest por defecto — ver
`testpaths` en `pyproject.toml` (solo `tests/`, `scripts/` nunca se
colecciona) y el guard de `BECARIO_LIVE_ROUTER_CHECK` en `main()`."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.live_router_check import (
    EditFixture,
    FixtureResult,
    ModelScore,
    RouteFixture,
    _parse_params,
    _parse_plan_context,
    _parse_step_list,
    build_scoreboard,
    load_fixtures,
    parse_fixture,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "router"


def _result(name: str, *, error=None, passes=3, latencies=(1.0, 1.0, 1.0)):
    return FixtureResult(
        name=name, passes=passes, attempts=3, latencies=latencies, error=error
    )


class TestParseStepList:
    def test_single_step(self):
        assert _parse_step_list("crear_directorio") == ["crear_directorio"]

    def test_multiple_steps_preserve_order(self):
        assert _parse_step_list("crear_directorio, listar_archivos") == [
            "crear_directorio", "listar_archivos",
        ]

    def test_empty_string_is_empty_list(self):
        assert _parse_step_list("") == []


class TestParseParams:
    def test_empty_string_is_empty_dict(self):
        assert _parse_params("") == {}

    def test_single_pair(self):
        assert _parse_params("particion=gpu") == {"particion": "gpu"}

    def test_multiple_pairs_preserve_all_keys(self):
        # Los valores numéricos se tipan: el router extrae params vía
        # Pydantic (nodos=4 es int) y la comparación debe ser homogénea.
        assert _parse_params("nodos=4,particion=gpu") == {
            "nodos": 4, "particion": "gpu",
        }

    def test_numeric_values_are_coerced(self):
        assert _parse_params("encut=520,parametro_red=3.23,formula=Zr") == {
            "encut": 520, "parametro_red": 3.23, "formula": "Zr",
        }


class TestParsePlanContext:
    def test_renders_the_same_numbered_shape_services_py_sends_the_llm(self):
        raw = "enviar_slurm(particion=default,nodos=1) | crear_directorio(destino_remoto=/a)"
        assert _parse_plan_context(raw) == (
            "1. enviar_slurm(particion=default,nodos=1)\n"
            "2. crear_directorio(destino_remoto=/a)"
        )

    def test_single_step_plan(self):
        assert _parse_plan_context("enviar_slurm(nodos=1)") == "1. enviar_slurm(nodos=1)"


class TestParseFixture:
    def test_route_fixture(self):
        text = (
            "kind: route\n"
            "prompt: creá /a y listá /b\n"
            "steps: crear_directorio, listar_archivos\n"
        )
        fx = parse_fixture(text, name="multi_demo.txt")
        assert isinstance(fx, RouteFixture)
        assert fx.prompt == "creá /a y listá /b"
        assert fx.expected_steps == ["crear_directorio", "listar_archivos"]

    def test_route_fixture_defaults_to_route_kind_when_missing(self):
        text = "prompt: creame la carpeta /a\nsteps: crear_directorio\n"
        fx = parse_fixture(text, name="single_demo.txt")
        assert isinstance(fx, RouteFixture)

    def test_edit_fixture_with_explicit_target(self):
        text = (
            "kind: edit\n"
            "plan: enviar_slurm(particion=default,nodos=1) | crear_directorio(destino_remoto=/a)\n"
            "prompt: paso 1: usá la partición gpu\n"
            "target_index: 1\n"
            "params: particion=gpu\n"
        )
        fx = parse_fixture(text, name="edit_demo.txt")
        assert isinstance(fx, EditFixture)
        assert fx.plan_context == (
            "1. enviar_slurm(particion=default,nodos=1)\n"
            "2. crear_directorio(destino_remoto=/a)"
        )
        assert fx.prompt == "paso 1: usá la partición gpu"
        assert fx.expected_target_index == 1
        assert fx.expected_params == {"particion": "gpu"}

    def test_edit_fixture_with_null_target_is_ambiguous_case(self):
        text = (
            "kind: edit\n"
            "plan: listar_archivos(destino_remoto=/a) | crear_directorio(destino_remoto=/b)\n"
            "prompt: cambiá algo\n"
            "target_index: null\n"
            "params:\n"
        )
        fx = parse_fixture(text, name="edit_ambiguous.txt")
        assert fx.expected_target_index is None
        assert fx.expected_params == {}


class TestLoadFixtures:
    def test_loads_every_txt_fixture_from_the_repo_directory(self):
        fixtures = load_fixtures(_FIXTURES_DIR)
        names = {fx.name for fx in fixtures}
        # Al menos un fixture de cada familia (SR8: single/multi/edit).
        assert any(n.startswith("single_") for n in names)
        assert any(n.startswith("multi_") for n in names)
        assert any(n.startswith("edit_") for n in names)

    def test_route_fixtures_have_at_least_one_expected_step(self):
        fixtures = load_fixtures(_FIXTURES_DIR)
        route_fixtures = [fx for fx in fixtures if isinstance(fx, RouteFixture)]
        assert route_fixtures
        for fx in route_fixtures:
            assert fx.expected_steps, f"{fx.name} no declara 'steps'"


class TestFixtureResult:
    def test_no_error_is_ok(self):
        assert _result("single_create_dir.txt").ok is True

    def test_error_is_not_ok(self):
        assert _result("x.txt", error="esperaba steps=[a]").ok is False

    def test_latency_is_the_median_of_the_attempts(self):
        # Mediana y no promedio: un outlier de scheduler no debe mover el
        # número (mismo criterio que la consola).
        assert _result("x.txt", latencies=(1.0, 2.0, 60.0)).latency_seconds == 2.0

    def test_to_dict_is_json_serializable(self):
        payload = _result("x.txt", error="rompió").to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["ok"] is False
        assert payload["error"] == "rompió"


class TestModelScore:
    def test_hits_counts_only_passing_fixtures(self):
        score = ModelScore(
            model="qwen2.5:7b",
            results=(_result("a.txt"), _result("b.txt", error="falló")),
        )
        assert (score.hits, score.total) == (1, 2)
        assert score.ok is False

    def test_all_passing_is_ok(self):
        score = ModelScore(model="m", results=(_result("a.txt"), _result("b.txt")))
        assert score.ok is True

    def test_skipped_model_is_ok_and_scores_zero_of_zero(self):
        # Un modelo no instalado no es un fallo de clasificación: la consola
        # imprime ⏭️ y el gate no lo cuenta. 0/0 salteado ≠ 0/6 real.
        score = ModelScore(model="gemma4:12b", skipped=True)
        assert (score.ok, score.hits, score.total) == (True, 0, 0)
        assert score.median_latency_seconds is None

    def test_median_latency_is_over_calls_not_over_fixture_medians(self):
        # Las seis llamadas son [1,1,1,10,20,30] -> mediana 5.5. Promediar
        # las medianas POR FIXTURE daría 10.5 y le daría el mismo peso a un
        # fixture rápido que a uno lento.
        score = ModelScore(
            model="m",
            results=(
                _result("rapido.txt", latencies=(1.0, 1.0, 1.0)),
                _result("lento.txt", latencies=(10.0, 20.0, 30.0)),
            ),
        )
        assert score.median_latency_seconds == 5.5

    def test_to_dict_is_json_serializable(self):
        payload = ModelScore(model="m", results=(_result("a.txt"),)).to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["hits"] == 1 and payload["total"] == 1


class TestBuildScoreboard:
    def _scoreboard(self):
        scores = [
            ModelScore(
                model="qwen2.5:7b",
                results=(_result("a.txt"), _result("b.txt", error="falló")),
            ),
            ModelScore(model="gemma4:12b", skipped=True),
        ]
        return build_scoreboard(scores, attempts=3, fixtures_dir=_FIXTURES_DIR)

    def test_records_what_was_measured_and_against_what(self):
        board = self._scoreboard()
        assert board["attempts"] == 3
        assert board["fixtures"] == ["a.txt", "b.txt"]
        assert [m["model"] for m in board["models"]] == ["qwen2.5:7b", "gemma4:12b"]

    def test_fixtures_dir_is_relative_to_the_repo_root(self):
        # El scoreboard se commitea: no puede filtrar la ruta absoluta de la
        # máquina que lo generó o el diff cambia según quién lo corra.
        assert self._scoreboard()["fixtures_dir"] == "tests/fixtures/router"

    def test_fixtures_dir_outside_the_repo_falls_back_to_its_own_path(self):
        board = build_scoreboard([], attempts=1, fixtures_dir=Path("/tmp/otros"))
        assert board["fixtures_dir"] == "/tmp/otros"

    def test_is_json_serializable(self):
        board = self._scoreboard()
        assert json.loads(json.dumps(board)) == board
