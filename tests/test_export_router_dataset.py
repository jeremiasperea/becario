"""Tests de `scripts/export_router_dataset.py`, el puente entre las
decisiones que un humano confirmó en producción y los fixtures del harness.

Todo sin red y sin base: `render_fixture` y `format_params` son puras, y
`export_fixtures` solo escribe archivos. El test que más importa es el
round-trip: lo que el exportador escribe lo tiene que poder leer
`scripts/live_router_check.py`, o los fixtures generados no miden nada."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_router_dataset import (
    export_fixtures,
    format_params,
    render_fixture,
)
from scripts.live_router_check import parse_fixture

_ROW_ID = 42


def _row(steps: list[dict], text: str = "preparar ZrO2", **overrides) -> dict:
    row = {
        "id": _ROW_ID,
        "text": text,
        "steps_json": json.dumps(steps, ensure_ascii=False),
        "model": "qwen2.5-coder:14b",
        "created_at": "2026-08-04T12:00:00+00:00",
        "outcome": "confirmed",
        "latency_seconds": 3.1,
    }
    row.update(overrides)
    return row


def _step(action: str = "preparar_calculo", **parametros) -> dict:
    return {"action": action, "parametros": parametros}


class TestFormatParams:
    def test_scalars_survive_sorted_by_key(self):
        assert format_params({"formula": "ZrO2", "encut": 520}) == (
            "encut=520, formula=ZrO2"
        )

    def test_floats_survive(self):
        assert format_params({"parametro_red": 5.07}) == "parametro_red=5.07"

    def test_lists_are_dropped_because_their_commas_break_the_format(self):
        # `puntos_k=[1,1,1]` se parsearía como tres pares rotos: el harness
        # corta por coma antes de mirar nada más.
        assert format_params({"puntos_k": [1, 1, 1], "formula": "Si"}) == "formula=Si"

    def test_dicts_are_dropped(self):
        assert format_params({"tags_incar": {"ENCUT": "600"}}) == ""

    def test_booleans_are_dropped(self):
        # Viajarían como "True" y el harness los leería como texto.
        assert format_params({"magnetico": True}) == ""

    def test_numeric_looking_strings_are_dropped(self):
        # El harness los coercionaría a int y la comparación daría ❌ falso.
        assert format_params({"nombre_trabajo": "123"}) == ""

    def test_values_with_separators_are_dropped(self):
        assert format_params({"nombre_trabajo": "a,b", "otro": "x=y"}) == ""

    def test_empty_values_are_dropped(self):
        assert format_params(
            {"formula": None, "red_cristalina": "", "miller": [], "tags_incar": {}}
        ) == ""

    def test_no_params_is_empty_string(self):
        assert format_params({}) == ""


class TestRenderFixture:
    def test_single_step_plan_carries_params(self):
        body = render_fixture(_row([_step(formula="ZrO2", encut=520)]))
        assert "prompt: preparar ZrO2\n" in body
        assert "steps: preparar_calculo\n" in body
        assert "params: encut=520, formula=ZrO2\n" in body

    def test_single_step_without_representable_params_omits_the_line(self):
        body = render_fixture(_row([_step("listar_archivos")]))
        assert "params:" not in body

    def test_multi_step_plan_never_carries_params(self):
        # `expected_params` del harness solo mira steps[0]: exportar los
        # params de un plan compuesto prometería cobertura que no existe.
        body = render_fixture(
            _row([_step(formula="Si"), _step("enviar_slurm", nodos=2)])
        )
        assert "steps: preparar_calculo, enviar_slurm\n" in body
        assert "params:" not in body

    def test_records_provenance_in_a_comment(self):
        body = render_fixture(_row([_step(formula="Si")]))
        assert body.startswith("# Confirmado por un humano el 2026-08-04")
        assert "qwen2.5-coder:14b" in body
        assert f"#{_ROW_ID}" in body

    def test_multiline_message_is_discarded(self):
        # Los fixtures se parsean línea a línea: este saldría corrupto.
        assert render_fixture(_row([_step()], text="preparar\nZrO2")) is None

    def test_blank_message_is_discarded(self):
        assert render_fixture(_row([_step()], text="   ")) is None

    def test_empty_plan_is_discarded(self):
        assert render_fixture(_row([])) is None


class TestRoundTripWithTheHarness:
    """Lo que escribe el exportador lo tiene que leer el harness."""

    def test_single_step_params_survive_the_round_trip(self):
        params = {"formula": "ZrO2", "encut": 520, "parametro_red": 5.07}
        body = render_fixture(_row([_step(**params)]))
        fx = parse_fixture(body, name="real_00042_preparar_zro2.txt")
        assert fx.prompt == "preparar ZrO2"
        assert fx.expected_steps == ["preparar_calculo"]
        # Mismos tipos, no solo mismo texto: el harness compara con `!=`
        # contra los params tipados que devuelve Pydantic.
        assert fx.expected_params == params

    def test_the_field_models_lose_is_the_one_that_survives(self):
        # `formula` y `red_cristalina` son los que los modelos locales
        # sueltan (0/3 medido). Si no sobrevivieran acá, todo esto no
        # mediría nada.
        body = render_fixture(
            _row([_step(formula="ZrO2", red_cristalina="fluorita")])
        )
        fx = parse_fixture(body, name="x.txt")
        assert fx.expected_params == {
            "formula": "ZrO2", "red_cristalina": "fluorita",
        }

    def test_multi_step_round_trips_with_no_expected_params(self):
        body = render_fixture(
            _row([_step(formula="Si"), _step("enviar_slurm", nodos=2)])
        )
        fx = parse_fixture(body, name="x.txt")
        assert fx.expected_steps == ["preparar_calculo", "enviar_slurm"]
        assert fx.expected_params == {}


class TestExportFixtures:
    def test_writes_only_confirmed_rows(self, tmp_path):
        rows = [
            _row([_step(formula="Si")], id=1),
            _row([_step(formula="Ge")], id=2, outcome="cancelled"),
        ]
        written, skipped = export_fixtures(rows, tmp_path)
        assert (written, skipped) == (1, 0)
        assert [p.name for p in tmp_path.iterdir()] == [
            "real_00001_preparar_zro2.txt"
        ]

    def test_counts_the_rows_it_could_not_represent(self, tmp_path):
        rows = [
            _row([_step(formula="Si")], id=1),
            _row([_step()], id=2, text="dos\nlíneas"),
        ]
        written, skipped = export_fixtures(rows, tmp_path)
        assert (written, skipped) == (1, 1)

    def test_creates_the_directory(self, tmp_path):
        out = tmp_path / "nuevo" / "router_real"
        export_fixtures([_row([_step(formula="Si")])], out)
        assert out.is_dir()

    def test_written_files_load_through_the_harness(self, tmp_path):
        from scripts.live_router_check import load_fixtures

        export_fixtures([_row([_step(formula="ZrO2", encut=520)])], tmp_path)
        fixtures = load_fixtures(tmp_path)
        assert len(fixtures) == 1
        assert fixtures[0].expected_params == {"formula": "ZrO2", "encut": 520}
