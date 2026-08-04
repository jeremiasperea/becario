# Comparación de modelos locales para el router

Benchmark de modelos servidos por Ollama para el enrutador de intenciones
de B.E.C.A.R.I.O. (ADR-0002: structured outputs; ADR-0006: plan-then-execute).
Fecha: 2026-07-17.

## Qué se mide y por qué

El router hace una sola llamada al LLM por mensaje: recibe texto libre en
español y devuelve un plan tipado (intención + parámetros) forzado por el
JSON Schema que Ollama aplica vía el parámetro `format`. El modelo no
redacta texto: clasifica y extrae. Por eso los dos ejes que deciden la
elección son:

1. **Correctitud**: acertar la secuencia de intenciones y los parámetros
   esperados, en especial en los planes multi-paso de ADR-0006.
2. **Latencia**: la máquina de desarrollo sirve el modelo en CPU pura
   (sin GPU), y el tiempo de generación es lo que el usuario de Telegram
   espera para que el bot le entienda.

## Método

- **Harness**: `scripts/live_router_check.py` contra un Ollama real en
  `localhost:11434`, con los 6 fixtures de `tests/fixtures/router/`
  (2 single-step, 2 multi-step, 2 de edición de plan).
- **Intentos**: 3 por fixture, decidido por mayoría (la generación en CPU
  no es bit-reproducible ni con `temperature=0`; ver docstring del
  harness). Un resultado sin corchetes de voto fue unánime (3/3).
- **Warm-up**: antes de medir cada modelo se hace una generación
  descartada, para que la carga del modelo a RAM no infle el primer
  fixture (el costo de carga crece con el tamaño del modelo y castigaría
  dos veces al más grande).
- **Latencia**: mediana de los 3 intentos por fixture (la mediana resiste
  outliers del scheduler); los intentos fallidos cuentan, porque un modelo
  que responde rápido pero mal no es más barato.
- **Aislamiento**: un solo modelo cargado por corrida (`ollama stop` del
  anterior antes de arrancar), sin otros procesos generando.

**Máquina**: WSL2, 20 cores, 23 GB RAM, sin GPU (CPU pura).
Ollama 0.31.2, contexto 4096.

## Resultados

Ordenados de más rápido a más lento por mediana de latencia.

| Fixture | qwen2.5:7b | gemma3:4b | gemma4:e4b | gemma4:12b |
|---|---|---|---|---|
| edit_explicit_step | ✅ 3.2s | ✅ 2.6s | ✅ 19.9s | ✅ 79.3s |
| edit_semantic_step | ✅ 3.1s | ✅ 2.6s | ✅ 26.2s | ✅ 67.6s |
| multi_destructive_tail | ✅ 9.1s | ✅ 11.9s | ✅ 28.0s | ❌ 0/3 · 149.9s |
| multi_two_safe_steps | ✅ 9.3s | ✅ 12.0s | ✅ 30.6s | ❌ 0/3 · 115.4s |
| single_create_dir | ✅ 5.1s | ✅ 9.4s | ✅ 14.5s | ✅ 70.0s |
| single_submit_job | ✅ 5.2s | ✅ 9.2s | ✅ 31.6s | ✅ 76.9s |
| **Aciertos** | **6/6** | **6/6** | **6/6** | **4/6** |
| **Mediana por llamada** | **5.4s** | **9.3s** | **27.0s** | **77.9s** |
| **Total (18 llamadas)** | **114s** | **148s** | **459s** | **1689s** |
| **Peso en RAM** | 4.7 GB | 3.3 GB | 9.5 GB | 8.9 GB |

## Hallazgos

- **`qwen2.5:7b` gana en los dos ejes**: 6/6 unánime y 5.4s de mediana —
  14× más rápido que `gemma4:12b` con la mitad de RAM. El tag `qwen2.5:7b`
  de Ollama ya es la variante instruct.
- **`gemma4:12b` falla la composición de forma sistemática**: en los dos
  fixtures multi-paso devolvió solo el primer paso del plan en los 9
  intentos (0/9). En esta máquina no es flakeo sino colapso: el
  plan-then-execute de ADR-0006 no funciona con este modelo. A 77.9s por
  mensaje tampoco es usable desde Telegram.
- **`gemma4:e4b` confirma su arquitectura "effective 4B"**: pesa MÁS en
  RAM que el 12b (9.5 vs 8.9 GB) pero genera ~3× más rápido, porque
  reduce cómputo, no memoria. Correcto en 6/6, pero sin ventaja frente a
  qwen en ningún eje.
- **`gemma3:4b` desmiente su etiqueta de "peor caso"**: el diseño lo trata
  como el modelo más chico y frágil (HC4), pero acertó 6/6 —incluidos los
  multi-paso que hunden al 12b— a 9.3s de mediana, y es el más liviano de
  todos en RAM (3.3 GB). Segundo en velocidad detrás de qwen. Un candidato
  serio si la RAM fuera el cuello de botella.
- **ADR-0002 pagó dividendos**: elegir structured outputs en vez de tool
  calling desacopló el proyecto de la familia Gemma. Cambiar de modelo
  fue cambiar `BECARIO_OLLAMA_MODEL`, sin tocar una línea de código.

## Limitaciones

- 6 fixtures no son un benchmark exhaustivo: cubren los caminos
  principales del router, no su distribución real de mensajes.
- Una sola máquina, CPU pura. En la máquina de producción (u otra con
  GPU) las latencias relativas pueden cambiar; la correctitud de
  composición del 12b merece re-verificarse allí antes de descartarlo.
- 3 intentos por fixture acotan el no-determinismo pero no lo eliminan.

## Recomendación

`BECARIO_OLLAMA_MODEL=qwen2.5:7b` en la máquina de desarrollo. Para
producción, correr este mismo harness en esa máquina antes de decidir.

## Segunda corrida (2026-08-04): qwen2.5-coder:14b duplica un paso

Con el harness ya emitiendo puntaje (`--json`), se midió el modelo que sirve
producción desde el benchmark del 2026-08-01 contra la línea base rápida. El
resultado crudo está en [`scoreboard_router.json`](scoreboard_router.json).

| | qwen2.5:7b | qwen2.5-coder:14b |
|---|---|---|
| **Aciertos** | **8/8** | **7/8** |
| **Mediana por llamada** | **8.8s** | **17.4s** |
| **Total (24 llamadas)** | 192s | 413s |

El único fallo, y no es flake —`[0/3]`, unánime:

```
❌ single_prepare_relax.txt — esperaba steps=['preparar_calculo'],
                              obtuve ['preparar_calculo', 'preparar_calculo']
```

**El 14b duplica el paso.** Dos `preparar_calculo` es una forma de plan
*válida* (no hay destructivos, entra en el rango 1-5), así que
`parse_llm_output` no la rechaza y el usuario termina con el cálculo
preparado dos veces.

Esto NO contradice el benchmark del 2026-08-01: aquel midió **invención** de
valores sobre 13 casos y el coder:14b ganó ahí. Este mide **forma del plan y
extracción de parámetros**, un eje que aquel no cubría. Son evidencias
complementarias sobre un modelo que cuesta el doble de tiempo por llamada.

Queda abierto: decidir si el 14b sigue sirviendo producción, y si la
duplicación se ataca desde el prompt o desde una validación de dominio que
rechace pasos idénticos consecutivos.

## Reproducir

```bash
# La corrida original de este documento
BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/live_router_check.py \
    --models gemma3:4b,qwen2.5:7b,gemma4:e4b,gemma4:12b --attempts 3 --timeout 300

# La segunda corrida, con scoreboard versionado
BECARIO_LIVE_ROUTER_CHECK=1 .venv/bin/python scripts/live_router_check.py \
    --models qwen2.5:7b,qwen2.5-coder:14b --timeout 300 \
    --json docs/scoreboard_router.json
```
