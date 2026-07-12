"""Tests del parseo de la salida del LLM con structured outputs."""
import json

from becario.domain.models import Intent
from becario.infrastructure.ollama_router import (
    OllamaRouter,
    RouterDecision,
    RouterParams,
)

parse = OllamaRouter.parse_llm_output

# Tamaño del schema de RouterDecision ANTES de fusionar `steps` (medido en
# la fase apply, tarea 3.1, sobre gemma3:4b como modelo peor-caso). El
# presupuesto del gate es 1.15x: si la fusión de `steps` en la tarea 3.2
# infla el schema por encima de eso, corresponde activar la mitigación de
# recorte de `description` descripta en el diseño (ADR-0006).
_ROUTER_DECISION_SCHEMA_BASELINE_BYTES = 3650
_ROUTER_DECISION_SCHEMA_BUDGET_FACTOR = 1.15


class TestParseLLMOutput:
    """`parse_llm_output` ahora devuelve un `Plan` (uno o más pasos). Un
    pedido de una sola acción sigue siendo un plan de un solo paso —
    `plan.single_step` expone ese paso sin cambiar el comportamiento de
    antes (SR1: paridad de regresión de un solo paso)."""

    def test_valid_action_with_params(self):
        plan = parse('{"steps": [{"action": "enviar_slurm", "parametros": {"nodos": 2}}]}')
        assert plan.is_single
        assert plan.single_step.action is Intent.SUBMIT_SLURM
        assert plan.single_step.parametros == {"nodos": 2}

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
            ("listar_archivos", Intent.LIST_FILES),
        ]:
            plan = parse(f'{{"steps": [{{"action": "{raw}", "parametros": {{}}}}]}}')
            assert plan.single_step.action is expected

    def test_structure_params_roundtrip(self):
        raw = (
            '{"steps": [{"action": "modificar_estructura", "parametros": '
            '{"formula": "Si", "supercelda": [2, 2, 2], "red_cristalina": "diamond"}}]}'
        )
        plan = parse(raw)
        assert plan.single_step.action is Intent.MODIFY_STRUCTURE
        assert plan.single_step.parametros["formula"] == "Si"
        assert plan.single_step.parametros["supercelda"] == [2, 2, 2]

    def test_calc_params_roundtrip(self):
        raw = (
            '{"steps": [{"action": "preparar_calculo", "parametros": '
            '{"formula": "Zr", "red_cristalina": "hcp", '
            '"tipo_calculo": "convergencia_encut", '
            '"encut_min": 250, "encut_max": 450, "encut_paso": 50}}]}'
        )
        plan = parse(raw)
        assert plan.single_step.action is Intent.PREPARE_CALC
        assert plan.single_step.parametros["tipo_calculo"] == "convergencia_encut"
        assert plan.single_step.parametros["encut_min"] == 250
        assert plan.single_step.parametros["encut_max"] == 450

    def test_none_params_are_excluded(self):
        plan = parse('{"steps": [{"action": "revisar_estado", "parametros": {"job_id": null}}]}')
        assert plan.single_step.parametros == {}

    def test_malformed_json(self):
        plan = parse("no soy json {")
        assert plan.is_single
        assert plan.single_step.action is Intent.UNKNOWN

    def test_empty_and_none(self):
        assert parse("").single_step.action is Intent.UNKNOWN
        assert parse(None).single_step.action is Intent.UNKNOWN

    def test_unknown_action_rejected_by_schema(self):
        plan = parse('{"steps": [{"action": "borrar_todo", "parametros": {}}]}')
        assert plan.single_step.action is Intent.UNKNOWN

    def test_missing_parametros_defaults_empty(self):
        plan = parse('{"steps": [{"action": "revisar_estado"}]}')
        assert plan.single_step.action is Intent.CHECK_STATUS
        assert plan.single_step.parametros == {}


class TestParseLLMOutputComposition:
    """Composición multi-paso (SR2): un pedido con varias acciones produce
    un `Plan` con varios pasos ordenados, cada uno con sus propios
    parámetros."""

    def test_two_safe_steps(self):
        raw = (
            '{"steps": ['
            '{"action": "crear_directorio", "parametros": {"destino_remoto": "/home/ana/a"}},'
            '{"action": "listar_archivos", "parametros": {"destino_remoto": "/home/ana/b"}}'
            ']}'
        )
        plan = parse(raw)
        assert not plan.is_single
        assert [s.action for s in plan.steps] == [Intent.CREATE_DIR, Intent.LIST_FILES]
        assert plan.steps[0].parametros == {"destino_remoto": "/home/ana/a"}
        assert plan.steps[1].parametros == {"destino_remoto": "/home/ana/b"}

    def test_three_steps_with_destructive_tail(self):
        raw = (
            '{"steps": ['
            '{"action": "crear_directorio", "parametros": {"destino_remoto": "/r"}},'
            '{"action": "modificar_estructura", "parametros": {"formula": "Si"}},'
            '{"action": "enviar_slurm", "parametros": {"script_remoto": "/r/run.sh"}}'
            ']}'
        )
        plan = parse(raw)
        assert len(plan.steps) == 3
        assert plan.steps[-1].action is Intent.SUBMIT_SLURM

    def test_plan_shape_violation_falls_back_to_unknown_single_step(self):
        # Dos pasos destructivos: viola la regla de plan (a lo sumo uno,
        # al final) -> el plan ENTERO se rechaza fail-closed (R1-003).
        raw = (
            '{"steps": ['
            '{"action": "enviar_slurm", "parametros": {}},'
            '{"action": "cancelar_calculo", "parametros": {"job_id": "1"}}'
            ']}'
        )
        plan = parse(raw)
        assert plan.is_single
        assert plan.single_step.action is Intent.UNKNOWN

    def test_destructive_step_not_last_falls_back_to_unknown(self):
        raw = (
            '{"steps": ['
            '{"action": "enviar_slurm", "parametros": {}},'
            '{"action": "listar_archivos", "parametros": {}}'
            ']}'
        )
        plan = parse(raw)
        assert plan.single_step.action is Intent.UNKNOWN

    def test_too_many_steps_falls_back_to_unknown(self):
        step = '{"action": "listar_archivos", "parametros": {}}'
        raw = '{"steps": [' + ",".join([step] * 6) + "]}"
        plan = parse(raw)
        assert plan.single_step.action is Intent.UNKNOWN


class TestSchema:
    def test_schema_restricts_actions_to_enum(self):
        schema = RouterDecision.model_json_schema()
        enum_values = schema["$defs"]["Intent"]["enum"]
        assert "enviar_slurm" in enum_values
        assert "cancelar_calculo" in enum_values
        assert "crear_directorio" in enum_values
        assert "listar_archivos" in enum_values

    def test_schema_has_structure_params(self):
        schema = RouterParams.model_json_schema()
        for key in ("formula", "supercelda", "red_cristalina", "parametro_red"):
            assert key in schema["properties"]

    def test_schema_has_calc_params(self):
        schema = RouterParams.model_json_schema()
        for key in ("tipo_calculo", "encut", "encut_min", "encut_max",
                    "encut_paso", "puntos_k"):
            assert key in schema["properties"]

    def test_schema_size_stays_within_budget(self):
        # Gate de tamaño (gemma3:4b es el peor caso): la fusión de `steps`
        # en RouterDecision (tarea 3.2) no puede inflar el schema más de
        # 1.15x el baseline medido antes del merge. Si esto falla, activar
        # la mitigación de recorte de `description` (ver diseño §2.2).
        size = len(json.dumps(RouterDecision.model_json_schema()))
        budget = _ROUTER_DECISION_SCHEMA_BASELINE_BYTES * _ROUTER_DECISION_SCHEMA_BUDGET_FACTOR
        assert size <= budget, (
            f"schema de RouterDecision creció a {size} bytes, "
            f"supera el presupuesto de {budget:.0f} bytes (1.15x baseline)"
        )

    def test_schema_steps_reuse_router_params_via_ref(self):
        # RouterParams NO debe duplicarse por paso: cada paso referencia
        # el mismo $def vía $ref (ver diseño §2.1).
        schema = RouterDecision.model_json_schema()
        assert "steps" in schema["properties"]
        assert schema["properties"]["steps"]["minItems"] == 1
        assert schema["properties"]["steps"]["maxItems"] == 5
        raw = json.dumps(schema)
        # RouterParams aparece definido una sola vez en $defs.
        assert raw.count('"RouterParams":') <= 1


class TestParseEditOutput:
    """`extract_edit` (tarea 5.1, diseño §3.1/§3.2): un cambio sobre un
    plan de VARIOS pasos trae, además del delta de parámetros, a qué paso
    (1-based, `target_index`) se refiere — semántico o explícito («paso
    N»). Fail-closed: cualquier salida fuera de schema se trata como "sin
    confianza" (`target_index=None`, sin delta) — nunca se adivina un
    paso."""

    parse_edit = staticmethod(OllamaRouter.parse_edit_output)

    def test_explicit_target_index_and_delta(self):
        decision = self.parse_edit('{"target_index": 2, "parametros": {"nodos": 4}}')
        assert decision.target_index == 2
        assert decision.parametros.nodos == 4

    def test_no_target_index_when_llm_is_not_confident(self):
        decision = self.parse_edit('{"target_index": null, "parametros": {"nodos": 4}}')
        assert decision.target_index is None
        assert decision.parametros.nodos == 4

    def test_missing_target_index_defaults_to_none(self):
        decision = self.parse_edit('{"parametros": {"particion": "gpu"}}')
        assert decision.target_index is None
        assert decision.parametros.particion == "gpu"

    def test_malformed_json_fails_closed(self):
        decision = self.parse_edit("no soy json {")
        assert decision.target_index is None
        assert decision.parametros.model_dump(exclude_none=True) == {}

    def test_empty_and_none_fail_closed(self):
        assert self.parse_edit("").target_index is None
        assert self.parse_edit(None).target_index is None
