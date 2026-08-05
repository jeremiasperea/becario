# ADR-0007: Confirmación de batch (todo plan con un cálculo)

> Se tomó para "planes grandes con varios cálculos"; desde la Enmienda 2
> (2026-08-05) cubre cualquier plan multi-paso que contenga un cálculo.

## Contexto

ADR-0006 (plan-then-execute) fija un plan de 1 a 5 pasos donde los pasos NO
destructivos se **auto-materializan** al construir el plan (crear carpeta,
subir, generar estructura) y solo la cola destructiva final pide
confirmación. El tope de 5 pasos existe para acotar el *blast-radius*: un
mensaje —posiblemente mal interpretado por el LLM o adversarial— no debería
disparar I/O sin límite sobre el cluster compartido, y con auto-materialización
cada paso es un efecto lateral sin confirmar.

El descompositor (ADR-0006 + trabajo posterior) parte un pedido compuesto
("relajá ZrO2 en rocksalt, zincblende y fluorite, cada una en su carpeta")
en instrucciones simples y rutea cada una. Pero cada instrucción de cálculo
con su carpeta rutea a `[preparar_calculo, crear_directorio]` = 2 pasos, así
que `mkdir + N cálculos` recompone a `1 + 2N` pasos: entra en 5 solo hasta
N=2 redes. Un pedido legítimo de 3+ redes se rechazaba —— el cap no distingue
entre un batch legítimo y un desborde por mala interpretación.

Además, el modelo de "cola de cálculos" emitía **una confirmación individual
por cálculo** (followups). Para un batch que el usuario pidió explícitamente
("las 3 redes"), N confirmaciones son el anti-patrón de aprobación en piloto
automático que el mismo ADR-0003 buscaba evitar.

## Decisión

> **Ampliado el 2026-08-05** — el disparador del batch cambió; ver
> "Enmienda 2" al final. Lo de abajo es la decisión como se tomó.

Un plan que excede la composición chica —— **más de
`_MAX_AUTOMATERIALIZE_STEPS` (5) pasos, O con varios `preparar_calculo`** ——
y está formado solo por pasos materializables y `preparar_calculo` (sin cola
realmente destructiva) es un **batch**: se confirma ENTERO, con una sola
confirmación, ANTES de tocar el cluster.

**Nada se auto-ejecuta.** `_prepare_batch` (application) valida cada cálculo
SIN I/O (`calc._build_calc_request`, sin generar ni subir), arma un preview
que enumera todos los pasos y grita el gasto de cómputo (N trabajos SLURM), y
stagea el plan entero como `PendingPlan(execute_all=True)`. Fail-closed: un
cálculo inválido aborta el batch sin efectos.

**Una sola confirmación autoriza todo.** Al confirmar, `_execute_batch` corre
TODOS los pasos en orden (`mkdir` → generar+subir+`sbatch` por cálculo, vía
`calc.execute_calc`), con corte al primer fallo y sin rollback (misma
semántica de ADR-0006, ahora sobre el plan entero). Cada paso se reporta
✅/❌/⏸. Un pedido explícito de N redes ES la autorización de N trabajos: la
seguridad viene del preview escaneable y una aprobación deliberada, no de
fragmentar en N confirmaciones.

**El cap deja de gatear el blast-radius.** Como en un batch nada se
auto-ejecuta (el humano ve el plan entero y lo puede rechazar antes de que
corra un solo `mkdir`), el argumento que justificaba el tope de 5 se disuelve.
`_MAX_PLAN_STEPS` sube a **11** y pasa a ser un tope ESTRUCTURAL: el
descompositor emite ≤5 instrucciones y cada una expande a ≤2 pasos, así que 11
acota una descomposición desbocada sin recortar un batch legítimo (hasta ~5
redes). El router directo sigue capado en 5 (`RouterDecision`); solo los planes
del descompositor llegan a 11.

**El camino chico no cambia.** Un plan de ≤5 pasos con a lo sumo un cálculo se
comporta byte a byte como en ADR-0006: prefijo auto-materializado + su
confirmación de cola (un solo cálculo mantiene su confirmación individual).

> ⚠️ **Este párrafo ya no describe el código (2026-08-05).** El camino chico
> con un cálculo SÍ cambió: hoy también es batch. Ver "Enmienda 2".

## Consecuencias

**A favor:** un pedido legítimo de 3+ redes ya no se rechaza; una sola
confirmación honesta reemplaza N aprobaciones en piloto automático; ningún
efecto toca el cluster hasta que el humano avala el batch entero (más seguro
que la auto-materialización previa, no menos).

**En contra:** en un batch, un fallo a mitad deja plantados los efectos de los
pasos ya ejecutados (consistente con ADR-0006, ahora sobre secuencias más
largas). Descubrir que un batch recompone a más de 11 pasos cuesta rutear
todas las partes antes de rechazarlo. No hay referencias cruzadas entre pasos
ni re-chequeo de duplicados por paso dentro del batch —— quedan para una
iteración futura.

**Enmienda a ADR-0006.** Queda superado el "límite documentado de v1:
`preparar_calculo` no se combina dentro de un plan de varios pasos" y la
confirmación individual por cálculo (followups) para planes multi-cálculo: ahora
esos planes son batches con una sola confirmación. El tope de 5 de ADR-0006 se
reinterpreta como umbral de auto-materialización (`_MAX_AUTOMATERIALIZE_STEPS`),
no como cap estructural del plan.

> ⚠️ **Esta enmienda prometió de más.** El límite de v1 quedó superado solo
> para los planes MULTI-cálculo, que son los que este ADR enrutaba al batch.
> Un plan con UN cálculo que no fuera el último —`[calc, crear_dir]`— siguió
> muriendo en el mensaje de ayuda genérico durante meses, mientras dos ADRs
> decían que el límite ya no existía. La Enmienda 2 lo cierra de verdad.

---

## Enmienda 2 (2026-08-05): el batch cubre cualquier plan con un cálculo

**Qué cambió.** El disparador ya no es el tamaño ni la cantidad de cálculos:
**todo plan multi-paso que contenga `preparar_calculo`** —en cualquier
posición— es un batch. Sigue exigiéndose que el resto sean pasos
materializables (sin cola realmente destructiva).

**Por qué.** El límite que regía no era el documentado. Medido antes de tocar
nada:

| forma | antes |
|---|---|
| `[crear_dir, calc]` | ✅ prefijo auto-materializado + confirmación del cálculo |
| `[calc, calc]` | ✅ batch (este ADR) |
| `[calc, crear_dir]` | ❌ mensaje de ayuda genérico |
| `[crear_dir, calc, listar]` | ❌ mensaje de ayuda genérico |

La POSICIÓN del cálculo decidía si el pedido vivía o moría. Y la "cola
destructiva virtual" que ADR-0006 parqueaba como trabajo futuro ya estaba
construida: es `_prepare_batch`, el mecanismo de este ADR. No hubo que
inventar nada, solo dejar de condicionarlo al tamaño.

**Lo que se pierde, y se eligió a sabiendas.** El párrafo "El camino chico no
cambia" deja de valer: `[crear_dir, calc]` ya **no** auto-materializa el
`mkdir`. Antes ese paso corría solo y un fallo aparecía en el mismo turno;
ahora el usuario ve el plan entero y recién al aprobar se ejecuta. Se eligió
porque *"nada pasó todavía"* es más fácil de razonar que *"algunos pasos
corrieron y otros no"*, y porque elimina el efecto huérfano de rechazar un
plan cuya mitad ya ocurrió. El costo es un turno más en los planes chicos.

**El preview dejó de ser cosmético.** Si nada se ejecuta hasta aprobar, esa
lista es lo ÚNICO sobre lo que la persona decide. Hubo que arreglar tres
cosas antes de poder activar el cambio:

1. `_describe_batch_step` describía solo `crear_directorio` y
   `preparar_calculo`; el resto salía como `• listar_archivos`, el nombre
   crudo del enum. Ahora cada intent dice qué va a hacer y con qué
   parámetros, y cuando el sistema resuelve un default lo nombra en vez de
   callarlo. Un test recorre `_MATERIALIZABLE_STEP_INTENTS` y falla si alguno
   cae en el fallback.
2. `_prepare_batch` validaba solo los cálculos. Un slab sin cara se stageaba
   igual y el usuario aprobaba un plan incumplible. Se extrajo
   `calc.validate_structure_params` —los chequeos de `modify_structure` que
   no tocan nada— y ahora la estructura recibe el mismo trato que el cálculo.
3. Contestar un dato faltante completaba SOLO el paso que lo pidió. En "armá
   el slab de ZrO2 y relajalo" —dos pasos sobre una superficie— el usuario
   contestaba «(001)» y el paso 2 volvía a preguntar lo mismo.
   `_propagate_answer` lleva el dato a los pasos posteriores del mismo
   material que no definan ya esa clave.

**Lo que sigue rechazándose.** Un cálculo junto a una cola destructiva
(`enviar_slurm`/`cancelar_calculo`): son dos confirmaciones en un turno, que
es justo el piloto automático que ADR-0003 y ADR-0006 evitan.

**Código retirado.** La "cola de cálculos" de `_run_composite_plan`
(materializar el prefijo y emitir un followup con confirmación individual por
cálculo) quedó inalcanzable: para llegar ahí un plan tendría que terminar en
cálculo Y tener un paso destructivo, y `_v_destructive_last` obliga al
destructivo a ir último. Verificado instrumentando la rama y corriendo la
suite entera: cero impactos. Con esto, la confirmación individual por cálculo
—que este ADR ya había reemplazado para los planes multi-cálculo— desaparece
del código, no solo de la doctrina.
