# ADR-0007: Confirmación de batch (planes grandes con varios cálculos)

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
