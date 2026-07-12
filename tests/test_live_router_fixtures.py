"""Tests de las funciones PURAS del harness de validación en vivo del
router (tarea 6.1, SR8). Estos tests corren siempre, sin red: validan el
parseo de los fixtures en `tests/fixtures/router/`. El harness en sí
(`scripts/live_router_check.py` invocado como script) SÍ pega contra un
Ollama real y está deliberadamente fuera del pytest por defecto — ver
`testpaths` en `pyproject.toml` (solo `tests/`, `scripts/` nunca se
colecciona) y el guard de `BECARIO_LIVE_ROUTER_CHECK` en `main()`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.live_router_check import (
    EditFixture,
    RouteFixture,
    _parse_params,
    _parse_plan_context,
    _parse_step_list,
    load_fixtures,
    parse_fixture,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "router"


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
        assert _parse_params("nodos=4,particion=gpu") == {
            "nodos": "4", "particion": "gpu",
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
