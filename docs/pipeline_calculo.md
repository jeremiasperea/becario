# Pipeline de un cálculo: de un mensaje de Telegram a una corrida VASP

Recorrido de extremo a extremo de lo que ocurre cuando un usuario pide un
cálculo DFT/VASP en lenguaje natural, hasta que los inputs quedan subidos
al cluster esperando confirmación de envío. Sigue la arquitectura
hexagonal del proyecto: cada etapa nombra su capa (dominio, aplicación,
puerto, adaptador) y los archivos reales que la implementan.

El mapa se derivó del grafo de conocimiento del repositorio
(`graphify-out/`) y se verificó contra el código.

## Vista general

```
Telegram (texto libre)
      │
      ▼  presentación
BecarioService.handle_text()                      application/services.py
      │
      ├─▶ IntentRouter.route(texto) ──────────────▶ Plan                (contrato de dominio)
      │        └─ segunda pasada: _backfill_structure() recupera formula/red
      │
      ├─▶ _maybe_decompose(texto, plan) ──────────▶ Plan (recompuesto)   (pedidos largos)
      │
      └─▶ _dispatch_plan(plan)
                 │  por cada PlanStep, según su Intent
                 ▼
        prepare_calc(svc, ctx, params)            application/handlers/calc.py
                 │
                 ├─▶ VaspCalcRequest(...)          (pedido validado; dominio)
                 │        └─ _v_formula · _v_crystal · scan_values (ENCUT)
                 │
                 ├─▶ CalcInputGenerator.generate() (PUERTO)  domain/ports.py
                 │        └─ VaspInputGenerator     (ADAPTADOR) infrastructure/vasp_inputs.py
                 │              ├─ make_bulk_atoms() → StructureResult   (ASE)
                 │              ├─ _render_incar() · _job_script()       (según CalcKind)
                 │              └─▶ CalcDirResult    (directorio local listo para subir)
                 │
                 ├─▶ ClusterGateway.upload_dir()   (PUERTO)  → SFTP al cluster
                 ├─▶ ClusterGateway.concat_files() (arma el POTCAR remoto)
                 │
                 └─▶ PendingPlan + PendingAction   → confirmación humana (token)
                          │
                          ▼  el usuario confirma
                 Intent.SUBMIT_SLURM → sbatch (ejecución real)
```

## Etapas

### 1. Entrada y ruteo — capa de aplicación

`BecarioService.handle_text()`
(`becario/application/services.py`) es la fachada. Llama a
`IntentRouter.route(texto)`, que devuelve un `Plan`: un plan tipado de 1 a
5 pasos, cada uno con su `Intent` y sus parámetros.

El router es un **adaptador** sobre Ollama con *structured outputs*
(`becario/infrastructure/ollama_router.py`, ADR-0002). Detrás del puerto
`IntentRouter` (`becario/domain/ports.py`) puede vivir cualquier modelo:
cambiar de modelo es cambiar `BECARIO_OLLAMA_MODEL`, sin tocar código.

**Segunda pasada de estructura.** Bajo el intent `preparar_calculo`, todos
los modelos locales medidos (qwen2.5:7b, gemma4:e4b, gemma3:4b) extraen los
parámetros numéricos (`tipo_calculo`, `encut_*`) pero sueltan `formula` y
`red_cristalina` — el bug es estructural, no del modelo ni del prompt (ver
`comparacion_modelos.md` y las fixtures `single_prepare_*`). Para cerrarlo,
`OllamaRouter.route()` completa el plan con `_backfill_structure()`: cuando
hay exactamente un paso `preparar_calculo` sin `formula`, hace una segunda
llamada enfocada (`extract_structure()`, sin *framing* de intent) que
recupera la estructura del material. El material es inequívoco con un solo
cálculo, aun cuando el paso venga acompañado de un `crear_directorio`.

### 2. Descomposición de pedidos largos — capa de aplicación

Los mensajes compuestos ("creá la carpeta X y relajá el bulk de X en bcc,
fcc y hcp") exceden el techo de extracción del modelo en una sola llamada.
`_maybe_decompose()` (`services.py`) detecta el caso — plan multi-paso con
algún `preparar_calculo` sin material — y lo reintenta vía
`IntentRouter.decompose()`: parte el pedido en instrucciones simples
auto-contenidas y rutea cada una por separado con `route()`. Como cada
`route()` aplica la segunda pasada de estructura, cada instrucción
descompuesta también recupera su `formula`/`red_cristalina`.

### 3. El contrato de dominio: `Plan`

`Plan` (`becario/domain/models.py`) es la *cintura angosta* de la
arquitectura: lo producen la infraestructura (`route`), lo declara el
puerto (`IntentRouter.route`) y lo orquesta la aplicación
(`_dispatch_plan`, `_run_composite_plan`), pero ninguna capa lo atraviesa.

Sus validadores corren **en orden de definición**, y el orden importa:

1. **`_v_merge_split_calc()`** — repara. Fusiona un cálculo que el modelo
   partió en dos mitades: la acción sin material (`preparar_calculo` sin
   `formula`) seguida INMEDIATAMENTE del material sin acción
   (`modificar_estructura` con `formula` y sin parámetros de archivo).
   `qwen2.5-coder:14b` parte así "relajá el bulk de W con red cristalina
   bcc", unánime en 3 de 3. Sin fusionar, el plan multi-paso con un cálculo
   se rechazaba fail-closed y el pedido moría en el mensaje de ayuda
   genérico. El ORDEN es lo que delata el error: armar la estructura
   DESPUÉS de calcular sobre ella no es un plan que alguien pida.
2. **`_v_collapse_stutter()`** — repara. Colapsa pasos consecutivos
   IDÉNTICOS Y COMPLETOS: el modelo a veces tartamudea y devuelve dos
   veces el mismo paso, que pasaba la validación de forma y hacía el
   trabajo dos veces. Solo fusiona si el paso dice sobre qué material
   trabaja: dos `preparar_calculo` sin `formula` pueden ser dos cálculos
   distintos a los que el modelo les comió el material, y fusionarlos se
   comería uno **y** borraría la señal de la que vive la recuperación por
   descomposición (§2).
3. **`_v_destructive_last()`** — rechaza. A lo sumo un paso destructivo y,
   si existe, debe ser el último, así la confirmación humana siempre
   gatilla sobre la cola irreversible del plan (ADR-0006).

Reparar antes de rechazar no es casual: un `enviar_slurm` tartamudeado son
DOS destructivos, y el validador de abajo tiraría el plan entero.
Colapsarlo primero lo deja en el único envío que el usuario pidió.

Y fusionar antes de colapsar tampoco: dos cálculos partidos ("relajá W bcc
y Si diamond", cuatro pasos) se vuelven dos cálculos completos y distintos,
que el colapso ya no toca. Al revés, las mitades sin material se verían
idénticas entre sí y se comerían un cálculo.

Nada de esto trunca ni reordena a espaldas del usuario: lo que se colapsa
es el mismo paso dicho dos veces, y lo que no encaja se rechaza ENTERO
antes de mostrar nada.

### 4. Preparación del cálculo — handler de aplicación

`_dispatch_plan()` enruta cada `PlanStep` a su handler. Para
`Intent.PREPARE_CALC` es `prepare_calc()`
(`becario/application/handlers/calc.py`). Sus pasos:

1. **Guardas de configuración y de entrada.** Si falta `formula`,
   repregunta el material (`calc.py:91`). Éste es el punto exacto donde la
   segunda pasada de la etapa 1 evita el rechazo: sin `formula`, el
   handler nunca avanza.
2. **Tipo de cálculo.** Resuelve `CalcKind` (relajación / estático /
   barrido de ENCUT). Si el modelo dio un rango de ENCUT pero omitió
   `tipo_calculo`, infiere `ENCUT_SCAN` (un rango solo tiene sentido como
   barrido).
3. **Pedido validado.** Construye `VaspCalcRequest`
   (`domain/models.py`), que valida `formula` (`_v_formula`, compartido con
   la validación de archivos), red cristalina (`_v_crystal`), supercelda,
   k-points y tiempo. El barrido de ENCUT se materializa con
   `scan_values()` / `default_encut_values()`.

### 5. Generación de inputs — puerto y adaptador

`prepare_calc` llama al puerto `CalcInputGenerator.generate(req)`
(`domain/ports.py`), implementado por `VaspInputGenerator`
(`becario/infrastructure/vasp_inputs.py`). El adaptador:

- Construye los átomos con `make_bulk_atoms()` vía el `StructureBuilder`
  de ASE (`infrastructure/ase_builder.py`), produciendo un
  `StructureResult`.
- Escribe los archivos VASP —`_render_incar()`, `_job_script()`,
  `_write_point()`, `_auto_kpoints()`—, todos gobernados por `CalcKind`:
  el tipo de cálculo decide qué INCAR y qué script SLURM se generan.
- Devuelve un `CalcDirResult`: el directorio de corrida local, listo para
  subir, con la lista de elementos en el orden de especies del POSCAR.

`CalcKind`, `StructureResult` y `CalcDirResult` (`domain/models.py`) son
los objetos de valor del dominio que fluyen entre handler, adaptador y
gateway. El dominio no conoce ASE ni SFTP: solo define los contratos.

### 6. Subida al cluster — puerto de infraestructura

Con el directorio local listo, `prepare_calc`:

- Resuelve los POTCAR: una variante por elemento, buscada en
  `BECARIO_POTCAR_DIR` (`ClusterGateway.file_exists`).
- Sube el directorio con `ClusterGateway.upload_dir()`
  (`domain/ports.py`) por SFTP, devolviendo un `CommandResult`.
- Arma el `POTCAR` remoto concatenando las variantes con
  `ClusterGateway.concat_files()`.

### 7. Confirmación y envío

La preparación **no envía** el trabajo: arma un `SlurmJobRequest`, calcula
una huella (`_calc_fingerprint`) para avisar duplicados, y deja el envío
en espera dentro de un `PendingPlan` con una `PendingAction`
(`Intent.SUBMIT_SLURM`). `_confirmations.put(plan)` devuelve un token y el
bot responde pidiendo confirmación. Recién cuando el usuario confirma se
ejecuta el `sbatch` real. La persistencia de la corrida queda a cargo de
`CalcRunRepository`.

## Mapa de capas

| Etapa | Capa | Archivo |
|---|---|---|
| `handle_text`, `_maybe_decompose`, `_dispatch_plan` | Aplicación (fachada) | `application/services.py` |
| `route`, `_backfill_structure`, `extract_structure`, `decompose` | Infraestructura (adaptador) | `infrastructure/ollama_router.py` |
| `IntentRouter`, `CalcInputGenerator`, `StructureBuilder`, `ClusterGateway` | Puerto (dominio) | `domain/ports.py` |
| `Plan`, `PlanStep`, `CalcKind`, `VaspCalcRequest`, `StructureResult`, `CalcDirResult` | Dominio (modelos) | `domain/models.py` |
| `prepare_calc`, `modify_structure` | Aplicación (handler) | `application/handlers/calc.py` |
| `VaspInputGenerator`, `make_bulk_atoms` | Infraestructura (adaptador) | `infrastructure/vasp_inputs.py`, `infrastructure/ase_builder.py` |
| `SSHClusterGateway.upload_dir` | Infraestructura (adaptador) | `infrastructure/ssh_gateway.py` |

## Por qué la segunda pasada está en el nacimiento del pipeline

`prepare_calc` aborta si falta `formula` (`calc.py:91`), y `VaspCalcRequest`
la vuelve a exigir en `_v_formula`. Sin `formula` y `red_cristalina`, no hay
`StructureResult`, no hay `CalcDirResult`, no hay nada que subir. La segunda
pasada de estructura (`_backfill_structure`) actúa en el primer eslabón —
la salida de `route()` — y desbloquea toda la cadena aguas abajo, tanto para
un cálculo simple como para cada instrucción de un pedido descompuesto.
