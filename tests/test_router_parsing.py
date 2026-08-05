"""Tests del parseo de la salida del LLM con structured outputs."""
import json

from becario.domain.models import Intent, Plan, PlanStep
from becario.infrastructure.ollama_router import (
    OllamaRouter,
    RouterDecision,
    RouterParams,
    compact_json_schema,
)

parse = OllamaRouter.parse_llm_output

# Techo del schema de RouterDecision, MEDIDO (2026-08-05,
# `scripts/calibrar_schema.py`). El presupuesto anterior era 3650 x 1.15 =
# 4197 B, y ninguno de los dos números era una medición: el 3650 se midió
# ANTES de fusionar `steps` y el 1.15 era headroom elegido para que ESE
# refactor (tarea 3.2) no regresara. Sirvió para lo que se escribió, pero
# quedó fosilizado como límite absoluto y llegó a estar al 98% de uso,
# contorsionando features que no hacía falta contorsionar.
#
# La calibración infló el schema con relleno inerte (los `title` que
# Pydantic autogenera y `compact_json_schema` poda) y corrió los 8 fixtures
# del harness en cada tamaño:
#
#     schema     qwen2.5:7b   gemma3:4b
#     4122 B        8/8          7/8
#     4884 B        8/8          7/8
#     6000 B        8/8          7/8
#     8004 B        8/8          7/8
#
# Ni precisión ni latencia se mueven al duplicar el schema. Y `gemma3:4b`
# —el peor caso que el presupuesto decía proteger— falla el MISMO fixture
# en todos los tamaños, incluido el actual: su fallo no es de tamaño.
#
# El techo queda en el mayor tamaño verificado limpio. El gate ya no
# raciona bytes: existe para cazar un crecimiento desbocado (p. ej. volver
# a un campo por cada tag del INCAR), no para decidir si un campo entra.
_ROUTER_DECISION_SCHEMA_MAX_BYTES = 8000


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
            ("ver_archivo", Intent.VIEW_FILE),
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

    def test_mp_source_params_roundtrip(self):
        raw = (
            '{"steps": [{"action": "preparar_calculo", "parametros": '
            '{"formula": "Fe2O3", "mp_id": "mp-19770", '
            '"fuente_estructura": "mp"}}]}'
        )
        plan = parse(raw)
        assert plan.single_step.action is Intent.PREPARE_CALC
        assert plan.single_step.parametros["mp_id"] == "mp-19770"
        assert plan.single_step.parametros["fuente_estructura"] == "mp"

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

    def test_schema_has_structure_source_params(self):
        schema = RouterParams.model_json_schema()
        for key in ("mp_id", "fuente_estructura"):
            assert key in schema["properties"]

    def test_schema_size_stays_within_measured_ceiling(self):
        # Gate contra crecimiento desbocado, no racionamiento de bytes: el
        # techo está MEDIDO (ver la nota del módulo y
        # `scripts/calibrar_schema.py`). Se mide el schema COMO VIAJA a
        # Ollama (compact_json_schema, sin los `title` autogenerados de
        # Pydantic) porque eso es lo que el modelo realmente recibe.
        size = len(json.dumps(compact_json_schema(RouterDecision)))
        assert size <= _ROUTER_DECISION_SCHEMA_MAX_BYTES, (
            f"schema de RouterDecision creció a {size} bytes, supera el techo "
            f"medido de {_ROUTER_DECISION_SCHEMA_MAX_BYTES} bytes. Antes de "
            f"subirlo, re-calibrá con scripts/calibrar_schema.py: el número "
            f"tiene que salir de una medición, no de una estimación."
        )

    def test_the_ceiling_leaves_real_room_for_new_features(self):
        # El presupuesto viejo (4197 B) llegó al 98% de uso y obligó a
        # contorsionar features. Este test falla si el margen vuelve a
        # apretarse hasta ahí, para que el aviso llegue antes de que alguien
        # vuelva a diseñar contra una pared.
        size = len(json.dumps(compact_json_schema(RouterDecision)))
        libre = _ROUTER_DECISION_SCHEMA_MAX_BYTES - size
        assert libre >= 1000, (
            f"quedan {libre} bytes libres de schema: volvimos a la situación "
            f"que motivó la calibración. Re-calibrá el techo con evidencia "
            f"antes de seguir agregando campos."
        )

    def test_compact_schema_has_no_titles_but_keeps_descriptions(self):
        # `title` es ruido para el LLM (el nombre ya está en la key) y
        # pesa contra el presupuesto; las `description` son la guía de
        # extracción y tienen que sobrevivir a la poda.
        schema = compact_json_schema(RouterDecision)
        raw = json.dumps(schema)
        assert '"title"' not in raw
        params = schema["$defs"]["RouterParams"]["properties"]
        assert "description" in params["destino_remoto"]
        assert "archivo" in params["destino_remoto"]["description"]

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


class TestBackfillStructure:
    """Segunda pasada de estructura: cuando el mensaje trae otro eje de
    extracción —el tipo de cálculo en `preparar_calculo`, la cara/capas en
    una losa— todos los modelos locales sueltan formula/red_cristalina
    (medido, model-agnóstico), aunque los extraen sin ese framing.
    `_backfill_structure` recupera el material cuando el plan tiene
    EXACTAMENTE un paso que lo necesita (donde es inequívoco, aunque venga
    con un crear_directorio); con dos o más manda el descompositor."""

    @staticmethod
    def _router(chat_stub):
        router = OllamaRouter()
        router._chat = chat_stub  # sin red: el backfill decide por el stub
        return router

    def test_recovers_structure_for_single_prepare(self):
        calls = []

        def chat(system_prompt, user_text, schema):
            calls.append(user_text)
            return '{"formula": "W", "red_cristalina": "bcc"}'

        router = self._router(chat)
        plan = Plan(steps=[PlanStep(
            action=Intent.PREPARE_CALC,
            parametros={"tipo_calculo": "relajacion"},
        )])
        result = router._backfill_structure("relajá el bulk de W bcc", plan)
        assert result.single_step.parametros == {
            "tipo_calculo": "relajacion", "formula": "W", "red_cristalina": "bcc",
        }
        assert calls == ["relajá el bulk de W bcc"]

    def test_router_extracted_value_wins_over_backfill(self):
        # El paso ya trae red_cristalina=fcc; el backfill no debe pisarlo.
        router = self._router(
            lambda s, u, sc: '{"formula": "W", "red_cristalina": "bcc"}'
        )
        plan = Plan(steps=[PlanStep(
            action=Intent.PREPARE_CALC,
            parametros={"red_cristalina": "fcc"},
        )])
        result = router._backfill_structure("x", plan)
        assert result.single_step.parametros["red_cristalina"] == "fcc"
        assert result.single_step.parametros["formula"] == "W"

    def test_noop_when_formula_already_present(self):
        calls = []
        router = self._router(lambda s, u, sc: calls.append(u) or "{}")
        plan = Plan(steps=[PlanStep(
            action=Intent.PREPARE_CALC, parametros={"formula": "Zr"},
        )])
        assert router._backfill_structure("x", plan) is plan
        assert calls == []  # sin segunda llamada

    def test_recovers_structure_for_single_modify(self):
        # El caso de la losa: el modelo se lleva la atención a la cara/capas
        # y suelta el material. Medido en los tres modelos locales.
        router = self._router(lambda s, u, sc: '{"formula": "ZrO2"}')
        plan = Plan(steps=[PlanStep(
            action=Intent.MODIFY_STRUCTURE,
            parametros={"tipo_estructura": "slab", "miller": [0, 0, 1], "capas": 5},
        )])
        result = router._backfill_structure("armá un slab de ZrO2 (001) de 5 capas", plan)
        assert result.single_step.parametros == {
            "tipo_estructura": "slab", "miller": [0, 0, 1], "capas": 5,
            "formula": "ZrO2",
        }

    def test_noop_for_intent_without_structure(self):
        calls = []
        router = self._router(lambda s, u, sc: calls.append(u) or "{}")
        plan = Plan(steps=[PlanStep(action=Intent.CREATE_DIR, parametros={})])
        assert router._backfill_structure("x", plan) is plan
        assert calls == []

    def test_noop_when_calc_and_structure_steps_coexist(self):
        # Dos pasos que necesitan material: cuál lleva cuál es ambiguo.
        calls = []
        router = self._router(lambda s, u, sc: calls.append(u) or "{}")
        plan = Plan(steps=[
            PlanStep(action=Intent.MODIFY_STRUCTURE, parametros={}),
            PlanStep(action=Intent.PREPARE_CALC, parametros={}),
        ])
        assert router._backfill_structure("x", plan) is plan
        assert calls == []

    def test_empty_crystal_is_dropped_not_backfilled(self):
        # El modelo emite "" (no null) para lo que no encontró, y un
        # crystal="" REBOTA en StructureRequest: colarlo cambiaría una
        # repregunta clara por un error de validación.
        router = self._router(lambda s, u, sc: '{"formula": "ZrO2", "red_cristalina": ""}')
        plan = Plan(steps=[PlanStep(action=Intent.MODIFY_STRUCTURE, parametros={})])
        result = router._backfill_structure("x", plan)
        assert result.single_step.parametros == {"formula": "ZrO2"}

    def test_invented_calc_params_are_filtered_out(self):
        # Medido: en pedidos de losa los modelos inventan encut/puntos_k/
        # tags_incar. La segunda pasada solo puede aportar el material.
        router = self._router(
            lambda s, u, sc: '{"formula": "Zr", "encut": 520, "puntos_k": [1, 1, 1],'
            ' "tags_incar": {"LORBIT": "-1"}}'
        )
        plan = Plan(steps=[PlanStep(action=Intent.MODIFY_STRUCTURE, parametros={})])
        result = router._backfill_structure("x", plan)
        assert result.single_step.parametros == {"formula": "Zr"}

    def test_recovers_structure_with_accompanying_create_dir(self):
        # Cada instrucción del descompositor rutea a [calc, crear_directorio]:
        # un solo cálculo -> material inequívoco -> se completa igual.
        router = self._router(
            lambda s, u, sc: '{"formula": "Zr", "red_cristalina": "bcc"}'
        )
        plan = Plan(steps=[
            PlanStep(action=Intent.PREPARE_CALC, parametros={"tipo_calculo": "relajacion"}),
            PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "Zr/bcc"}),
        ])
        result = router._backfill_structure("relajá Zr bcc en Zr/bcc", plan)
        assert result.steps[0].parametros == {
            "tipo_calculo": "relajacion", "formula": "Zr", "red_cristalina": "bcc",
        }
        # El paso no-cálculo queda intacto y el orden se preserva.
        assert result.steps[1].parametros == {"destino_remoto": "Zr/bcc"}
        assert [s.action for s in result.steps] == [Intent.PREPARE_CALC, Intent.CREATE_DIR]

    def test_noop_for_multiple_calc_steps(self):
        # Dos cálculos: material ambiguo, no se adivina -> lo parte el
        # descompositor. Sin segunda llamada.
        calls = []
        router = self._router(lambda s, u, sc: calls.append(u) or "{}")
        plan = Plan(steps=[
            PlanStep(action=Intent.PREPARE_CALC, parametros={"tipo_calculo": "relajacion"}),
            PlanStep(action=Intent.PREPARE_CALC, parametros={"tipo_calculo": "estatico"}),
        ])
        assert router._backfill_structure("x", plan) is plan
        assert calls == []

    def test_fail_open_when_second_pass_empty(self):
        # Ollama caído / fuera de schema: se conserva el plan sin material.
        router = self._router(lambda s, u, sc: None)
        plan = Plan(steps=[PlanStep(action=Intent.PREPARE_CALC, parametros={})])
        assert router._backfill_structure("x", plan) is plan
