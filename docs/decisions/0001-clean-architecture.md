# ADR-0001: Clean Architecture con puertos y adaptadores

## Contexto

B.E.C.A.R.I.O. integra varios sistemas externos con ciclos de cambio y
naturaleza muy distinta: la API de Telegram, un LLM local servido por
Ollama, conexiones SSH a un cluster HPC, y una base SQLite. La primera
versión (un workflow de n8n) mezclaba la lógica de negocio con el detalle
de cada integración en el mismo nodo, lo que hacía imposible testear la
sanitización de comandos sin desplegar el workflow completo.

## Decisión

Organizar el código en capas concéntricas — `domain`, `application`,
`infrastructure`, `presentation` — donde las dependencias solo apuntan
hacia adentro. El dominio define entidades validadas (Pydantic) y puertos
(`Protocol`) sin importar nada de Telegram, paramiko u Ollama. La capa de
aplicación (`BecarioService`) orquesta casos de uso contra esos puertos.
La infraestructura implementa los puertos con librerías concretas. Un
único módulo (`main.py`) conoce todas las implementaciones y las cablea
(inyección de dependencias manual, sin framework).

## Consecuencias

**A favor:** el 100% de la lógica de negocio y de sanitización se testea
sin red, sin SSH y sin Telegram (ver `tests/test_service.py`,
`tests/test_models.py`). Cambiar Ollama por otro LLM, o SQLite por
Postgres, no toca el dominio ni la aplicación.

**En contra:** más archivos e indirección que un script único; para un
prototipo descartable sería sobre-ingeniería. Se justifica porque el
software es de uso continuo por un grupo de investigación y es el
artefacto que este trabajo describe y justifica.
