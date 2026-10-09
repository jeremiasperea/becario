"""Contrato entre los enums de dominio y el vocabulario que ve el router.

Las descripciones de campo y el prompt se derivan de los enums; estos tests
fallan si un miembro nuevo no llega al texto que recibe el LLM.
"""

import pytest

from becario.domain.models import (
    Axis,
    CalcKind,
    OutputFormat,
    StructureKind,
    StructureSource,
)
from becario.infrastructure.ollama_router import (
    _GLOSAS_CALCULO,
    _SYSTEM_PROMPT,
    RouterParams,
)

_PROPIEDADES = RouterParams.model_json_schema()["properties"]

_CAMPOS = [
    ("tipo_estructura", list(StructureKind)),
    ("eje_vacio", list(Axis)),
    ("formato_salida", list(OutputFormat)),
    ("fuente_estructura", [f for f in StructureSource if f is not StructureSource.AUTO]),
    ("tipo_calculo", list(CalcKind)),
]


@pytest.mark.parametrize(
    "campo, miembro",
    [(c, m) for c, ms in _CAMPOS for m in ms],
    ids=lambda x: getattr(x, "value", x),
)
def test_cada_valor_esta_en_la_descripcion_del_campo(campo, miembro):
    assert miembro.value in _PROPIEDADES[campo]["description"]


def test_glosas_cubren_exactamente_calc_kind():
    assert set(_GLOSAS_CALCULO) == set(CalcKind)


@pytest.mark.parametrize("kind", list(CalcKind), ids=lambda k: k.value)
def test_cada_tipo_de_calculo_esta_en_el_prompt(kind):
    assert f"'{kind.value}' ({_GLOSAS_CALCULO[kind]})" in _SYSTEM_PROMPT
