"""Contrato entre el enum `Intent`, el prompt del router y los handlers.

Cada acción que el router puede decidir tiene que estar descrita en el
prompt (si no, el LLM nunca la elige) y tener un handler en el servicio
(si no, el `.get()` devuelve None en silencio). `UNKNOWN` queda afuera:
se resuelve como rechazo, no como acción.
"""

import pytest

from becario.application.services import BecarioService
from becario.domain.models import Intent
from becario.infrastructure.ollama_router import _SYSTEM_PROMPT

_ACCIONES = [i for i in Intent if i is not Intent.UNKNOWN]


@pytest.mark.parametrize("intent", _ACCIONES, ids=lambda i: i.value)
def test_cada_accion_esta_descrita_en_el_prompt(intent):
    assert f"'{intent.value}'" in _SYSTEM_PROMPT


@pytest.mark.parametrize("intent", _ACCIONES, ids=lambda i: i.value)
def test_cada_accion_tiene_handler(intent):
    # `_intent_handlers` solo arma `partial`s sobre `self`, no lo usa:
    # alcanza con un objeto cualquiera para leer la tabla.
    assert intent in BecarioService._intent_handlers(object())
