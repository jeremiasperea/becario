"""Tests del parseo de la salida del LLM con structured outputs."""
from becario.domain.models import Intent
from becario.infrastructure.ollama_router import (
    OllamaRouter,
    RouterDecision,
    RouterParams,
)

parse = OllamaRouter.parse_llm_output


class TestParseLLMOutput:
    def test_valid_action_with_params(self):
        routed = parse('{"action": "enviar_slurm", "parametros": {"nodos": 2}}')
        assert routed.intent is Intent.SUBMIT_SLURM
        assert routed.params == {"nodos": 2}

    def test_all_known_actions(self):
        for raw, expected in [
            ("modificar_estructura", Intent.MODIFY_STRUCTURE),
            ("preparar_calculo", Intent.PREPARE_CALC),
            ("enviar_slurm", Intent.SUBMIT_SLURM),
            ("consultar_db", Intent.QUERY_DB),
            ("consultar_resultados", Intent.QUERY_RESULTS),
            ("revisar_estado", Intent.CHECK_STATUS),
            ("cancelar_calculo", Intent.CANCEL_JOB),
            ("crear_directorio", Intent.CREATE_DIR),
        ]:
            routed = parse(f'{{"action": "{raw}", "parametros": {{}}}}')
            assert routed.intent is expected

    def test_structure_params_roundtrip(self):
        raw = (
            '{"action": "modificar_estructura", "parametros": '
            '{"formula": "Si", "supercelda": [2, 2, 2], "red_cristalina": "diamond"}}'
        )
        routed = parse(raw)
        assert routed.intent is Intent.MODIFY_STRUCTURE
        assert routed.params["formula"] == "Si"
        assert routed.params["supercelda"] == [2, 2, 2]

    def test_calc_params_roundtrip(self):
        raw = (
            '{"action": "preparar_calculo", "parametros": '
            '{"formula": "Zr", "red_cristalina": "hcp", '
            '"tipo_calculo": "convergencia_encut", '
            '"encut_min": 250, "encut_max": 450, "encut_paso": 50}}'
        )
        routed = parse(raw)
        assert routed.intent is Intent.PREPARE_CALC
        assert routed.params["tipo_calculo"] == "convergencia_encut"
        assert routed.params["encut_min"] == 250
        assert routed.params["encut_max"] == 450

    def test_none_params_are_excluded(self):
        routed = parse('{"action": "revisar_estado", "parametros": {"job_id": null}}')
        assert routed.params == {}

    def test_malformed_json(self):
        assert parse("no soy json {").intent is Intent.UNKNOWN

    def test_empty_and_none(self):
        assert parse("").intent is Intent.UNKNOWN
        assert parse(None).intent is Intent.UNKNOWN

    def test_unknown_action_rejected_by_schema(self):
        assert parse('{"action": "borrar_todo", "parametros": {}}').intent is Intent.UNKNOWN

    def test_missing_parametros_defaults_empty(self):
        routed = parse('{"action": "revisar_estado"}')
        assert routed.intent is Intent.CHECK_STATUS
        assert routed.params == {}


class TestSchema:
    def test_schema_restricts_actions_to_enum(self):
        schema = RouterDecision.model_json_schema()
        enum_values = schema["$defs"]["Intent"]["enum"]
        assert "enviar_slurm" in enum_values
        assert "cancelar_calculo" in enum_values
        assert "crear_directorio" in enum_values

    def test_schema_has_structure_params(self):
        schema = RouterParams.model_json_schema()
        for key in ("formula", "supercelda", "red_cristalina", "parametro_red"):
            assert key in schema["properties"]

    def test_schema_has_calc_params(self):
        schema = RouterParams.model_json_schema()
        for key in ("tipo_calculo", "encut", "encut_min", "encut_max",
                    "encut_paso", "puntos_k"):
            assert key in schema["properties"]
