# ADR-0002: Structured outputs en vez de tool calling

## Contexto

El bot necesita traducir texto libre a una acción estructurada (intención +
parámetros). La forma estándar de resolver esto con LLMs modernos es *tool
calling*: el modelo recibe un catálogo de herramientas con schema y
devuelve una invocación tipada. Ollama expone tool calling para modelos
como Llama 3.1 o Qwen 2.5, pero **no** para la familia Gemma, que es el
modelo elegido para este proyecto (`gemma4:12b`, servido localmente).

## Decisión

Usar *structured outputs*: el parámetro `format` de la API de Ollama
acepta un JSON Schema arbitrario y obliga al modelo a responder JSON
conforme a ese schema, sin importar si el modelo soporta tool calling.
El schema se deriva automáticamente de un modelo Pydantic
(`RouterDecision` en `infrastructure/ollama_router.py`), así que hay una
sola fuente de verdad entre "lo que le pedimos al LLM" y "lo que el
código espera recibir".

## Consecuencias

**A favor:** funciona con cualquier modelo servido por Ollama, no ata el
proyecto a una familia de modelos con tool calling nativo. Agregar un
parámetro nuevo es agregar un campo a `RouterParams`; el contrato con el
LLM se actualiza solo.

**En contra:** no hay ejecución de herramientas en el mismo turno (el LLM
no puede "encadenar" varias llamadas como en un agente ReAct de tool
calling real) — el enrutador sigue siendo de un solo turno: un mensaje,
una intención. Encadenar pasos (p. ej. "generá la estructura y después
enviá el cálculo") queda para una capa de planificación futura, no
resuelta por este mecanismo.

**Nota de robustez:** aun con structured outputs, el parseo final sigue
siendo defensivo (`RouterDecision.model_validate_json`) — un modelo local
mal configurado, cortado a mitad de generación, o corriendo una versión
de Ollama sin soporte pleno de `format`, puede devolver JSON inválido
igual. Nunca se asume que la garantía del proveedor es absoluta.
