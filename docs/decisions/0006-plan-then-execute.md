# ADR-0006: Composición de pasos (plan-then-execute)

## Contexto

El enrutador (ADR-0002) es de un solo turno: un mensaje produce una sola
`RoutedRequest` (intención + parámetros). Esa decisión dejó explícitamente
afuera "encadenar pasos (p. ej. «generá la estructura y después enviá el
cálculo»)... para una capa de planificación futura, no resuelta por este
mecanismo". En uso real, buena parte de los pedidos del grupo SÍ son
compuestos: "creame la carpeta X y después subí Y", "generá el POSCAR de
Si y creá la carpeta de la corrida". Hoy eso obliga a mandar dos mensajes
separados, y el usuario no tiene forma de pedir en un solo turno una
secuencia con sentido.

Encadenar pasos con un loop tipo ReAct (una llamada al LLM por paso,
decidiendo sobre la marcha) multiplica la latencia por la cantidad de
pasos y abre una superficie nueva: cada llamada intermedia es una
oportunidad más para que el modelo alucine o se desvíe del pedido
original. Y el mecanismo de confirmación humana de ADR-0003 fue diseñado
para UNA acción destructiva por turno — extenderlo a "N confirmaciones,
una por paso" degrada la experiencia sin agregar seguridad real (el
usuario termina aprobando en piloto automático).

## Decisión

Una sola llamada de structured output devuelve un **plan**: `steps:
list[PlanStep]` (mínimo 1, máximo 5 pasos), cada uno con su propia
`action` y `parametros` — un pedido de una sola acción sigue siendo,
técnicamente, un plan de un solo paso, así que el camino de hoy (una
llamada al LLM, una confirmación, una ejecución) queda intacto byte a
byte; no hay dos code paths distintos según la cantidad de pasos, solo
uno generalizado.

**Invariante destructivo-al-final.** El dominio (`Plan._v_destructive_last`
en `domain/models.py`) valida que a lo sumo un paso sea destructivo
(`enviar_slurm`/`cancelar_calculo`) y, si existe, que sea el ÚLTIMO paso
del plan. Un plan que viola esto (dos pasos destructivos, o uno que no
está al final) se rechaza ENTERO antes de mostrar nada — nunca se trunca,
reordena ni se ejecuta parcialmente. Esto generaliza ADR-0003
directamente: la confirmación humana sigue gatillando sobre una sola
operación irreversible por turno, ahora al final de una secuencia en vez
de ser la secuencia entera.

**Una sola confirmación, al final.** Los pasos no destructivos (crear
carpeta, listar archivos, generar una estructura) se materializan EN
ORDEN al construir el plan, igual que ya hacían solos hoy — no esperan a
que el usuario confirme nada, porque no son irreversibles. Si el plan
termina en un paso destructivo, recién ahí se arma la confirmación, y
solo para esa cola; el resto ya corrió y se muestra como ejecutado
(`✅`/`❌`/`⏸`). Confirmar ejecuta la cola destructiva UNA vez (mismo
token de un solo uso de ADR-0003); rechazar deja plantados los efectos
ya materializados — es el mismo comportamiento que ya tenía hoy un
`preparar_calculo` rechazado (sube los inputs y después el usuario no
confirma el envío), no una regresión nueva.

**Corte en el primer fallo, sin rollback.** La ejecución es estrictamente
en orden; si un paso falla, el plan se detiene ahí — los pasos
siguientes se reportan "no intentados", los ya ejecutados NO se
deshacen. No hay rollback automático a propósito: `mkdir`/subir un
archivo/`sbatch` no son operaciones transaccionales entre sí (no existe
una forma genérica y segura de "deshacer" un `mkdir` remoto sin arriesgar
borrar algo que no correspondía), y fingir una garantía de rollback que
no se puede sostener sería peor que no ofrecerla: el usuario vería un
mensaje de éxito falso sobre el estado del cluster.

**Tope de 5 pasos.** Hoy el router emite como mucho una intención por
mensaje, así que un mensaje dispara como mucho un efecto lateral (una
subida, un `mkdir`, una escritura local). Componer pasos multiplica eso:
N pasos son hasta N efectos laterales auto-ejecutados desde un solo
mensaje. `_MAX_PLAN_STEPS = 5` acota esa amplificación a una composición
chica y humanamente plausible (crear carpeta + armar estructura +
preparar cálculo, o dos lecturas + un envío) sin abrir la puerta a que un
mensaje mal interpretado —o adversarial— dispare I/O sin límite sobre el
cluster.

**Edición = re-preparación completa, nunca parche silencioso.** El botón
✏️ Modificar (ADR existente, ahora generalizado) targetea el paso
correcto de un plan de varios pasos con una sola llamada extra al LLM:
semánticamente ("usá la partición gpu" apunta al paso que envía o
calcula, no al que crea una carpeta) o explícitamente ("paso 2: ..."). Si
el modelo no tiene confianza sobre a qué paso se refiere el cambio, el
plan se re-presenta SIN TOCAR y se refresca el TTL — nunca se fusiona ni
se ejecuta a ciegas (mismo principio fail-safe que ya regía la extracción
de un solo paso). Un cambio aceptado no parchea el `payload` ya
materializado de un paso: reconstruye el plan entero desde los pedidos
ORIGINALES de cada paso (el editado, con el delta fusionado; el resto,
sin cambios) y lo vuelve a correr por la MISMA ruta de construcción de un
plan nuevo — así la confirmación que ve el usuario siempre coincide con
lo que realmente se re-subió/re-generó, nunca con un estado a mitad de
camino.

## Consecuencias

**A favor:** un pedido compuesto legítimo ("creá la carpeta y corré el
script") ya no exige dos mensajes; el modelo de seguridad de ADR-0003 no
se diluye (sigue habiendo una sola operación irreversible confirmada por
turno); el camino de un solo paso es indistinguible del de antes en
latencia, UX y validación (una sola llamada al LLM, mismos botones,
mismo texto).

**En contra:** un plan rechazado deja plantados los efectos de los pasos
no destructivos que ya corrieron (no hay forma de "deshacer" un `mkdir`
sin riesgo); esto es consistente con el comportamiento previo de
`preparar_calculo`, pero ahora aplica también a composiciones más largas.
No hay referencias cruzadas entre pasos (el paso 2 no puede usar el
resultado del paso 1) ni re-chequeo de duplicados por paso dentro de un
plan compuesto — quedan para una iteración futura.

**Límite de v1 (SUPERADO, ver abajo): `preparar_calculo` no se combina
dentro de un plan de varios pasos.** `preparar_calculo` deja su propia
confirmación pendiente (sube inputs, arma el POTCAR, prepara un
`SlurmJobRequest`) — es, en los hechos, una operación "casi destructiva"
que el validador de `Plan` no conoce (`Intent.destructive()` solo cubre
`enviar_slurm` y `cancelar_calculo`). Dejarlo entrar como paso "seguro"
de un plan compuesto auto-ejecutaría esos efectos sin pasar por la
confirmación de un solo paso que tiene hoy, o exigiría inventar un
segundo mecanismo de confirmación-dentro-de-un-plan que no está diseñado.
La implementación rechazaba fail-closed cualquier plan compuesto que
contuviera `preparar_calculo` (nada se ejecuta, se responde el mensaje de
ayuda genérico), y quedaba anotada para una futura iteración una "cola
destructiva virtual" para operaciones que preparan y confirman sin ser
técnicamente `Intent.destructive()`.

**Consecuencia medida (2026-08-05): ese fail-closed convierte un error de
formato del modelo en un pedido muerto.** `qwen2.5-coder:14b` —el modelo
de producción— parte "relajá el bulk de W con red cristalina bcc" en dos
mitades, unánime en 3 de 3 intentos:

```
paso 1: preparar_calculo     {tipo_calculo: relajacion}
paso 2: modificar_estructura {formula: W, red_cristalina: bcc}
```

La acción sin material, el material sin acción. Ninguna se sostiene sola,
y como el plan es multi-paso y contiene `preparar_calculo`, se rechaza
ENTERO. Corrido por el camino real del servicio, el usuario recibe:

```
❓ No pude interpretar tu pedido. Probá con:
• "Relajá los parámetros de red del bulk de W"
```

La primera sugerencia es lo que acaba de pedir. El pedido más común del
proyecto, roto en el modelo de producción, con un mensaje que se burla
del usuario sin querer.

La red de recuperación existente no alcanza:
`BecarioService._needs_decomposition` detecta la firma (plan multi-paso
con un cálculo sin material) y vuelve a rutear descompuesto, pero el
modelo devuelve el MISMO plan partido y el fallback es el plan roto
original. `OllamaRouter._backfill_structure` tampoco: exige exactamente
un paso que necesite material, y acá hay dos (`preparar_calculo` y
`modificar_estructura` están los dos en `_STRUCTURE_INTENTS`), así que se
abre de manos justo en el caso que podría arreglar.

**Se arregla en el dominio** (`Plan._v_merge_split_calc`), no en el
prompt. Es el reverso de `_v_collapse_stutter`: allá el modelo dice dos
veces un paso completo, acá dice una sola vez un paso partido en dos.
Las condiciones son estrechas a propósito — el cálculo sin material, la
estructura inmediatamente después y con material, y sin parámetros de
archivo (`destino_remoto`, `formato_salida`, `nombre_archivo`), porque
"generá el POSCAR de Si en /home/ana" es un entregable propio. El orden
es lo que delata el error: armar la estructura DESPUÉS de calcular sobre
ella no es un plan que alguien pida.

**Por qué NO en el prompt, con evidencia.** Se intentó primero: una regla
explícita ("un cálculo es UN SOLO paso, el material va en el mismo paso")
más un ejemplo con otro material que el del fixture, para no enseñarle al
modelo la respuesta del examen. Medido: `coder:14b` siguió partiendo el
pedido **3 de 3**, idéntico. La regla se revirtió en vez de dejarla
puesta. Prosa que está medida como inerte no es una red de seguridad: es
un amuleto, y encima uno que no puede vigilar CI. La regla de dominio sí
se testea sin LLM.

Todo lo anterior describe el estado **mientras el fail-closed regía**. Ese
episodio fue lo que empujó a mirar el límite en sí, que es lo que sigue: la
reparación de dominio arregla ESTA forma de error, y el levantamiento del
límite hace que la próxima forma no sea fatal.

**Levantamiento del límite (2026-08-05): la cola destructiva virtual es
el batch, y ya existía.** (El detalle de este levantamiento vive en la
Enmienda 2 de ADR-0007, que es el ADR dueño del batch; acá queda el efecto
sobre el límite de v1.) ADR-0007 construyó exactamente el mecanismo que
faltaba —validar y describir cada paso sin tocar el cluster, stagear el
plan entero con UNA confirmación— pero solo se usaba para planes grandes
o multi-cálculo. Medido antes de tocar nada, el límite real que regía no
era el documentado:

| forma | antes |
|---|---|
| `[crear_dir, calc]` | ✅ andaba (prefijo auto-materializado + confirmación del cálculo) |
| `[estructura, calc]` | ✅ andaba |
| `[calc, calc]` | ✅ andaba (batch) |
| `[calc, crear_dir]` | ❌ mensaje de ayuda genérico |
| `[calc, listar]` | ❌ mensaje de ayuda genérico |
| `[crear_dir, calc, listar]` | ❌ mensaje de ayuda genérico |

El límite no era "`preparar_calculo` no se combina" sino "`preparar_calculo`
tiene que ser el ÚLTIMO". La posición de un paso decidía si el pedido
vivía o moría, y el ADR describía el estado anterior a ADR-0007.

**Ahora todo plan multi-paso que contenga `preparar_calculo` se confirma
entero como batch**, en cualquier posición. Dos consecuencias, y la
segunda es un cambio de contrato:

1. Los `[calc, …]` dejan de morir.
2. Los `[crear_dir, calc]` **ya no auto-materializan el prefijo**: antes
   el `mkdir` corría solo y el usuario se enteraba de un fallo en el mismo
   turno; ahora ve el plan completo y recién al aprobar se ejecuta. Se
   eligió a propósito: "algunos pasos ya corrieron y otros no" es más
   difícil de razonar que "nada pasó todavía", y elimina el efecto
   huérfano de rechazar un plan cuya mitad ya ocurrió.

Sigue rechazándose fail-closed **un cálculo junto a una cola destructiva**
(`enviar_slurm`/`cancelar_calculo`): son dos confirmaciones en un turno,
que es justo la degradación que este ADR evita a propósito.

**Lo que el cambio obligó a arreglar antes.** Si nada se ejecuta hasta
aprobar, el preview deja de ser cosmético y pasa a ser el contrato sobre
el que la persona decide:

- `_describe_batch_step` describía solo `crear_directorio` y
  `preparar_calculo`; el resto salía como `• listar_archivos`, el nombre
  crudo del enum. Ahora cada intent dice qué va a hacer, con sus
  parámetros y —cuando el sistema resuelve un default— cuál es ese
  default. Un test recorre `_MATERIALIZABLE_STEP_INTENTS` y falla si
  alguno cae en el fallback del enum.
- El batch validaba solo los cálculos antes de stagear. Un slab sin cara
  se stageaba igual y el usuario aprobaba un plan incumplible. Se extrajo
  `calc.validate_structure_params` (los chequeos de `modify_structure` que
  no tocan nada) y el batch la corre por cada paso de estructura: mismo
  trato que el cálculo.
- Contestar un dato faltante completaba SOLO el paso que lo pidió. En
  "armá el slab de ZrO2 y relajalo" —dos pasos sobre una sola superficie—
  el usuario contestaba «(001)» y el paso 2 volvía a preguntar lo mismo
  con las mismas palabras. El bug era anterior (verificado corriendo el
  mismo escenario contra `main`), pero quedaba tapado porque el paso 1 sí
  se ejecutaba y se veía progreso; sin materialización previa quedaba a la
  vista y se leía como que el bot ignoró la respuesta.
  `_propagate_answer` lleva el dato a los pasos POSTERIORES del MISMO
  material que no definan ya esa clave.

**Código muerto retirado.** La "cola de cálculos" de
`_run_composite_plan` (materializar el prefijo y emitir un followup con
confirmación individual por cálculo) quedó inalcanzable: para llegar ahí
un plan tendría que terminar en cálculo y a la vez tener un paso
destructivo, y `_v_destructive_last` obliga al destructivo a ir último.
Verificado además instrumentando la rama y corriendo la suite entera: cero
impactos. Se sacó en vez de dejarla — código muerto que describe un
comportamiento que ya no existe se lee como si rigiera.

**Nota de presupuesto de schema:** fusionar `steps` en `RouterDecision`
crece el JSON Schema que se le manda al LLM (gemma3:4b es el peor caso —
el modelo de banco, ver ADR-0002). Medido en la fase de implementación:
baseline pre-fusión **3650 bytes**; primer intento de fusión **4406
bytes** (por encima del presupuesto de 1.15× = 4197 bytes); tras recortar
los docstrings de `RouterStep`/`RouterDecision` (Pydantic los vuelca como
`description` en el schema — texto de prompt, no de validación) **3897
bytes**, dentro de presupuesto. `RouterParams` se referencia una sola vez
vía `$ref` en `$defs` y se reutiliza por paso — no se duplica el schema
de parámetros por cada paso del plan. Un test de regresión
(`tests/test_router_parsing.py::TestSchema::test_schema_size_stays_within_measured_ceiling`)
falla si una fusión futura vuelve a inflar el schema por encima del
1.15×.

**Actualización — el presupuesto está prácticamente agotado.** Las features
posteriores (losas con índice de Miller, fuente `relajado`, tags del INCAR
a mano, tipo de cálculo `dos`) sumaron campos y el schema pasó de los
**3897 bytes** medidos acá a **4117**: quedan ~80 de los 4197.

La conclusión de diseño, que vale más que el número: **un campo del router
por cada feature no escala.** Ese patrón funciona dos o tres veces y
después no hay de dónde sacar bytes. Por eso los tags del INCAR entraron
como UN campo genérico (`tags_incar`, un diccionario) validado contra el
vocabulario del manual, y no como un campo por tag — cuesta 126 bytes una
sola vez y sirve para los 169 tags documentados.

**Actualización 2 (2026-08-05) — el presupuesto era falso, y por eso
"agotarlo" no significaba nada.** Con el tablero del router andando (PR
\#31) se pudo hacer lo que faltaba desde el principio: medir. El resultado
está en `scripts/calibrar_schema.py`, que infla el schema con relleno
inerte (los `title` que Pydantic autogenera y `compact_json_schema` poda) y
corre los 8 fixtures en cada tamaño:

| schema | qwen2.5:7b | gemma3:4b |
|---|---|---|
| 4122 B (el de hoy) | 8/8 | 7/8 |
| 4884 B | 8/8 | 7/8 |
| 6000 B | 8/8 | 7/8 |
| **8004 B** | **8/8** | **7/8** |

Ni la precisión ni la latencia se mueven al **duplicar** el schema. Y
`gemma3:4b` —el peor caso que este presupuesto decía proteger— falla el
mismo fixture (`single_prepare_encut`) en TODOS los tamaños, incluido el
actual: su fallo no es de tamaño.

De dónde venía el error: ni el 3650 ni el 1.15 fueron nunca una medición
de capacidad del modelo. El 3650 se midió **antes** de fusionar `steps` y
el 15% era headroom para que ESE refactor no regresara — un guard de
regresión puntual, correcto para lo que se escribió. Lo que falló fue
reinterpretarlo como límite absoluto y diseñar contra él durante meses.

Dos cosas más que conviene dejar dichas:

- **El gate vigilaba el 40% del costo.** Por cada `route()` el modelo
  recibe el schema (4122 B) **y** `_SYSTEM_PROMPT` (6179 B). El consejo de
  arriba —"si hace falta un campo nuevo, va sin `description`, la guía vive
  en `_SYSTEM_PROMPT`, que no pesa contra este presupuesto"— no ahorraba
  contexto: movía bytes de la columna medida a la no medida. Era
  contabilidad, no ahorro.
- **El techo nuevo (8000 B) es el mayor tamaño verificado limpio**, y el
  gate cambió de propósito: ya no raciona bytes para decidir si un campo
  entra, sino que caza un crecimiento desbocado (volver a un campo por tag
  del INCAR, por ejemplo). Quedan ~3900 bytes libres.

Lo que **sí** sigue en pie de la conclusión anterior: un campo por feature
no escala como diseño, y `tags_incar` como diccionario genérico sigue
siendo la decisión correcta. Lo que cae es la urgencia y, sobre todo,
recortar `description` para ahorrar bytes — las descripciones son guía de
extracción y ahora se pagan sin problema.

**Limitación de la calibración:** ningún relleno es perfectamente inerte,
así que el resultado se lee como cota. Aguantar 8 KB de relleno prueba que
el tamaño por sí solo no degrada hasta ahí; no prueba dónde está el
límite exacto ni cómo interactúan campos con significado real. Antes de
subir el techo de nuevo, re-correr `scripts/calibrar_schema.py`.

**Nota de validación en vivo (AR-2): `gemma4:12b` verificado, con dos
hallazgos.** La paridad de fixtures sobre el modelo de producción (SR8,
diferida en el archive como AR-2) se corrió y pasa 6/6. La corrida dejó
dos hallazgos que condicionan cómo interpretar el harness
(`scripts/live_router_check.py`):

1. *Rutas de una sola letra colapsan el plan.* Con paths abstractos tipo
   `/home/ana/x` y `/home/ana/y`, `gemma4:12b` descarta determinísticamente
   el segundo paso y emite un plan de un solo paso (reproducido 5/5;
   también con `x`/`z`, lo que descarta la colisión con la conjunción
   "y"). Con rutas realistas compone correctamente (3/3). No es una falla
   de composición del modelo sino sensibilidad a placeholders
   adversariales: los fixtures usan rutas realistas y quedó anotado en
   `multi_two_safe_steps.txt` que no se vuelva a placeholders de una
   letra. `gemma3:4b` no exhibe esta sensibilidad.

2. *Sobre CPU la inferencia no es determinística ni con
   `temperature=0`.* El decoding greedy no es bit-reproducible entre
   threads (el orden de reducción de floats varía), y `gemma4:12b` queda
   cerca del borde de decisión "1 paso vs 2 pasos": flakea ~1 de cada 6
   corridas en composición multi-paso aun con el mismo input. Además una
   generación del 12b en CPU puede superar los 120 s del timeout por
   defecto del router y producir ❌ espurios. Por eso el harness decide
   cada fixture por MAYORÍA sobre `--attempts` intentos (default 3; el
   voto se imprime solo si no fue unánime) y acepta `--timeout` para
   subir el tope por request. Un ❌ aislado del harness en 12b/CPU no es
   evidencia de regresión por sí solo; una mayoría fallida sí.
