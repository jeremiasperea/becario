# Plan de refactor del bot

Este plan no salió de leer el código buscando cosas feas. Salió de dos cosas
puestas una al lado de la otra:

1. **La bitácora real.** 118 mensajes en `chat_messages`, del 18/07 al 04/08
   de 2026, con los pedidos que un usuario efectivamente le hizo al bot por
   Telegram.
2. **La misma bitácora corrida hoy**, contra el cluster de prueba, Ollama y
   Materials Project reales, con la batería de `tests/conversaciones/`
   (27 escenarios, `scripts/replay_conversaciones.py`).

Esa segunda parte es la que cambia el plan. **De las fallas que la bitácora
mostraba, la mayoría ya están arregladas.** Un plan escrito solo sobre el log
habría mandado a arreglar seis cosas que andan.

## Los números

De los 49 pedidos en lenguaje natural de la bitácora (54 turnos del usuario,
5 de ellos botones), **24 terminaron mal**: 22 con error o aviso, 2 sin
ninguna respuesta. Casi la mitad. Y los pedidos que más se repiten son
exactamente los que fallaban — nadie repite seis veces por gusto:

| Veces | Pedido |
|---|---|
| 6 | `relajá el ZrO2` |
| 5 | `quiero que crees una carpeta llamada ZrO2 … bcc, fcc y hcp` |
| 4 | `listá mi home` |
| 4 | `Armá un slab de ZrO2 (001) de 5 capas, 2x1` |
| 3 | `mostramelo en forma de tree` |

Corrida hoy, la batería da **22/27** (1078 s, `qwen2.5-coder:14b`).

## Qué se arregló solo (y por qué importa decirlo)

Estos pedidos fallaban en julio y **hoy pasan**:

| Escenario | Fallaba con | Hoy |
|---|---|---|
| CV09 / CV10 | carpeta + 3 relajaciones: los cálculos perdían la fórmula | ✅ 7 pasos, `formula` en los tres |
| CV12 | `buscalos en poject matirials` → derivaba ZrO2 a Zr | ✅ |
| CV13 | slab de ZrO2 → error crudo de ASE | ✅ repregunta clara |
| CV14 | pregunta por el parámetro de red → error de constructor | ✅ |
| CV15 | `fluorita a=5.07` → convertía el material en CaF₂ | ✅ |
| CV06 | `mostramelo en forma de tree` → «Ruta inválida: '/'» | ✅ árbol |
| CV08 | tipeo `quiero crees` → creaba `/Zr/bcc` en la raíz | ✅ relativo |
| CV25 | doble ✏️ → decía «expiró» teniendo el plan vivo | ✅ |
| CV26 | `subí el ENCUT a 600` → volcaba el `ValidationError` de pydantic | ✅ |

Esto no es un detalle de proceso. Significa que **el trabajo pendiente es
mucho más chico de lo que el log sugiere**, y que meterse a rediseñar el
descompositor o la capa de errores hoy sería tocar código que funciona.

## Lo que sigue roto

Cinco escenarios en rojo, tres defectos reales.

### D1 — «Mi home» no existe en el modelo de rutas (CV02, CV03)

Reproducido hoy, igual que en julio:

```
vos> listá mi home
bot> 📂 /data/becario_runs_qa/20260807_014321:
```

`_resolve_workspace_path` (`remote_files.py:51-76`) conoce dos casos:
absoluta pasa tal cual, relativa se ancla al `remote_base`. **El home del
usuario no es representable.** No hay nada que el router pueda emitir que
funcione: si manda una ruta la inventa, y si no manda nada cae en el
`remote_base`.

En julio el modelo llegó a emitir `destino_remoto=/home/ana` — un usuario que
no existe en el cluster. No es casual: `/home/ana/…` aparece **8 veces** en
`_SYSTEM_PROMPT` (`ollama_router.py:257-285`) como ruta de ejemplo, y las
líneas 287-288 agregan *«jamás las copies a tu respuesta»*. Una instrucción
negativa no borra un ejemplo del contexto: lo subraya.

El gateway **ya tiene** `home_dir()` — se usa en la línea 68 para anclar un
`remote_base` relativo. El dato está; falta la forma de pedirlo.

### D2 — Una pregunta es siempre una orden (CV22)

Reproducido hoy, textual:

```
vos> donde buscaste?
bot> 📊 Estado trabajos (root):
     JOBID  NAME  STATE  TIME PARTITION
```

Todo mensaje tiene que mapear a una acción sobre el cluster; no hay intención
para «explicame lo que acabás de decir». Lo que más molesta es que la
respuesta estaba en el mensaje anterior del propio bot (`/data/potcars`,
variantes `O_sv, O_pv, O`): no hacía falta ni el LLM ni el cluster.

En la bitácora pasa lo mismo con «segun Proyect marials cual es el parametro
de red para ZrO2 tetragonal» (CV14), que hoy ya no revienta pero tampoco
responde la pregunta.

### D3 — El bot pregunta sin escuchar (CV11)

```
vos> relajá el bulk
bot> ⚠️ Decime qué material querés calcular…     ← ok=False, sin pendiente
```

Es una pregunta que no deja nada esperando. La respuesta funciona igual
porque se rutea de cero, así que en una conversación de a uno no se nota.
Se nota en un plan de varios cálculos: en la bitácora (mensajes 47-50) esa
misma pregunta salió **tres veces seguidas** y ninguna de las tres esperaba
a nadie, así que contestar una sola vez no podía completar el pedido.

Compárese con el camino de compuesto, que sí arma pendiente
(`awaiting_params=True`). Dos preguntas del mismo handler, con memoria
distinta.

### Nota sobre CV24 — falso positivo mío

El primer toque de ✏️ devuelve `ok` sin `awaiting_params`; el segundo
(`ALREADY_MODIFYING_TEXT`, `services.py:701`) sí lo trae. El pendiente queda
armado igual —CV26 lo demuestra—, así que es una incoherencia interna, no un
defecto de cara al usuario. La aserción era mía y la corregí.

## El defecto que la batería no ve

### D4 — El aviso de vencimiento está muerto en producción

`telegram_bot.py:236`:

```python
chat, self._service.start_modification, token,
requester_id=query.from_user.id, chat_id=chat,   # chat es un telegram.Chat
```

`start_modification(..., chat_id: int)` guarda ese valor en
`_PendingEdit.chat_id`, y `sweep_expired_pendings()` se lo pasa a
`bot.send_message(chat_id=...)` (`telegram_bot.py:266`). Un objeto `Chat`
donde va un entero: la llamada falla, el `except` de la línea 268 se la come
y la loguea, y **el aviso nunca llega**.

Ese aviso es una funcionalidad que el README documenta y justifica en detalle
(línea 68). Los tests están en verde: los diez tests de `start_modification`
pasan `chat_id=1`. **La suite prueba la función; nadie prueba la llamada.**
La batería tampoco: llama al servicio, igual que los tests. Es el costo
exacto de tener la frontera presentación↔aplicación cubierta solo con dobles.

## El riesgo estructural

### D5 — El estado de la conversación es lo único que no se persiste

`InMemoryConfirmationStore` (`storage.py:286`) y `_pending_edits`
(`services.py:172`, un `dict`) son memoria del proceso. El historial, los
trabajos, la bitácora, las decisiones del router y las corridas están todos
en SQLite. **Lo único volátil es justo lo que hace que una conversación sea
una conversación.**

La batería no lo detecta —corre en un proceso continuo— y por eso CV17, CV19,
CV20 y CV25 pasan. Ese contraste es la evidencia:

- **Hoy, en un proceso vivo:** `relajá el ZrO2` → repregunta de fase →
  `tetragonal` → tarjeta de confirmación. Perfecto, cuatro variantes de
  tipeo incluidas.
- **En julio, en producción:** mensaje 99 (15:10) pregunta la fase, mensaje
  100 (15:24) contesta `tetragonal` — **13 minutos, con TTL de 30** — y
  recibe *«No pude interpretar tu pedido»*. Y el mensaje 106 tocó ✏️ **47
  segundos** después de la tarjeta, con TTL de 600, y recibió *«expiró o ya
  fue usada»*.

Ninguno de los dos venció por tiempo. El proceso se reinició en el medio.

Hay un agravante: el paliativo de `services.py:214-222` reemplaza el
`HELP_TEXT` por un aviso útil **solo si hay un pendiente vencido**. Un
pendiente que se fue con el proceso no deja rastro, así que no se puede
distinguir de un mensaje que de verdad no se entendió. El código no puede
dar la respuesta correcta porque ya no tiene el dato.

Esto además bloquea correr el bot como servicio con `Restart=on-failure`
(la unidad systemd del README): cada reinicio le cuesta una conversación al
usuario, en silencio.

## Corrección sobre el diagnóstico de fallos

**Acá me equivoqué y conviene dejarlo escrito**, porque la conclusión
opuesta parecía obvia.

Cuando el trabajo 11 falló, el bot dijo:

> «lo más probable es que el script nunca llegara a ejecutarse en el nodo
> (¿el directorio es visible desde los nodos de cómputo?)»

Yo lo di por errado: `sacct` reportaba `ExitCode 127` («command not found»),
y verifiqué que `vasp_std` no está en el `PATH` de los workers. Parecía
cerrado.

No lo era. `/root/becario_runs` **no es compartido**: el controlador y el
worker ven contenidos distintos. En el nodo, `slurm-11.out` dice:

```
bash: /root/…/Zr_relajacion_20260802_234857/run_vasp.sh: No such file or directory
```

El script se subió al `/root` del controlador y el nodo tenía el suyo, vacío.
**El diagnóstico del bot era correcto.**

De ahí salen dos cosas, y ninguna es la que yo había escrito:

- **D6 — Nadie verifica que `remote_base` sea compartido.** El bot manda
  trabajos a un directorio que los nodos de cómputo no ven, y se entera por
  el fracaso. Es un chequeo de arranque de una sola vez, en la misma línea
  fail-fast que ya se aplica a Ollama y al token (ADR-1/ADR-2). Con
  `BECARIO_REMOTE_BASE=/data/becario_runs` (compartido) el envío de hoy
  llegó al nodo y corrió — falló por otra cosa, ver abajo.
- **D7 — El diagnóstico acierta por inferencia, no por evidencia.** Llega a
  la conclusión correcta desde la *ausencia* de archivos, y la enuncia como
  «lo más probable». El código de salida —127, que `job_status`
  (`ssh_gateway.py:167`) **ya pide** en su `--format`— no se consulta: el
  monitor usa `job_state`, que trae solo `State`. Con esa línea, el aviso
  puede decir el código en vez de conjeturar.

Y una confirmación aparte, del envío que hizo la batería hoy (CV23, job 1,
`ExitCode 127`, en `/data` que **sí** es compartido):

```
vasp.out: run_vasp.sh: line 4: vasp_std: command not found
```

El cluster de prueba no tiene VASP en el `PATH` (el binario está en
`/data/Vasp.5.4.4/bin` y `BECARIO_VASP_PRELUDE` está vacío). Es config del
entorno de prueba, no del bot — pero explica por qué ningún trabajo de la
bitácora terminó bien, y conviene arreglarlo antes de creerle a una corrida.

---

## Medición: ¿router en dos etapas?

Se evaluó partir el router en dos pasadas (clasificar la familia, después
extraer con un schema chico y afinado). `scripts/medir_schemas_router.py`,
18 pedidos reales de la bitácora × 3 repeticiones × 3 brazos,
`qwen2.5-coder:14b`. Detalle en `docs/medicion_schemas.json`.

| | A schema grande | B schema chico | C chico + lo que faltaba |
|---|---|---|---|
| **sistema** (rutas inventadas) | **0/30** · 11.2 s | 15/30 · 8.8 s | **0/30**, ancla 30/30 · 10.1 s |
| **cálculo** (fórmulas erradas) | 18/24 · 14.2 s | **0/24** · 13.3 s | **0/24** · **4.9 s** |

**La respuesta no es la misma para las dos familias, y ese es el hallazgo.**

- **Rutas: el prompt grande gana.** El brazo B no solo empeora, reproduce
  literalmente los bugs de julio — `/home/usuario`, `/corridas/Zr`,
  `/Zr/bcc` en la raíz, 3 de 3 cada uno. Los 6454 caracteres no son grasa:
  cargan la convención de que una ruta relativa ancla en corridas, y el
  prompt chico la perdió. Lo que sí paga es el enum `base`
  (home/corridas/absoluta): 30/30 sin alucinar, y **más rápido que A**. O
  sea que D1 se arregla agregando vocabulario al schema actual, sin segunda
  etapa.
- **Material: el prompt grande estorba.** Pierde `formula` en 6 de los 7
  pedidos reales; solo sobrevive `Generá un POSCAR de Si diamond 2x2x2`. El
  `_STRUCT_PROMPT` de producción (600 ch) acierta 24/24 y tarda **un tercio**
  (4.9 s vs 14.2 s).

Ojo con leer la fila de cálculo como si fuera producción: hoy `route()`
aplica `_backfill_structure`, que **es** el brazo C. El 18/24 es el piso del
schema grande solo, y la medida de cuánto trabajo está haciendo esa red de
seguridad — que resultó ser casi todo.

**Conclusión:** no partir el router en dos etapas. La clasificación ya acierta
42/44 en la bitácora, y una pasada extra costaría latencia en todos los
mensajes para arreglar un 5% que no está ahí. Lo que sí sale de la medición:

1. Agregar `base` al schema actual (es D1, ahora con evidencia).
2. Dejar de tratar el backfill como red de seguridad: en material es el
   camino principal, es 3× más rápido y no falla. Hoy solo se dispara cuando
   hay exactamente un paso de estructura (`ollama_router.py:571`).

---

## Estado de ejecución

Al 2026-08-09. La suite pasó de **943 a 972 tests** y la batería de
conversaciones de **22/27 a 28/28**. Todo verde.

| Fase | Estado | Qué se hizo |
|---|---|---|
| 0 | ✅ | D4 arreglado (`chat.id`, no el objeto) + tests de frontera |
| 1 | ✅ | `SQLiteConfirmationStore` y `SQLitePendingEditStore`, cableados |
| 2 | ✅ | D3 (las repreguntas esperan) y D2 (intención `explicar`) |
| 3 | ✅ | enum `base` en el schema + D7 adelantado (dos desvíos, ver abajo) |

Escenarios que cambiaron de estado:

| | Antes | Ahora |
|---|---|---|
| CV02, CV03 — «listá mi home» | ❌ | ✅ |
| CV11 — contestar el material que faltaba | ❌ | ✅ |
| CV22 — «donde buscaste?» | ❌ | ✅ |
| CV24 — modificar antes de confirmar | ❌ (aserción mía) | ✅ |
| CV28 — el pendiente sobrevive un reinicio | no existía | ✅ |

Sin regresiones en ningún escenario. El tablero del router
(`docs/scoreboard_router.json`) quedó re-medido en 8/8 con
`qwen2.5-coder:14b`.

Tres cosas que salieron de ejecutarlo y no estaban en el diagnóstico:

- **Un bug latente en los dos stores de confirmación.** `pop()` de un plan
  VENCIDO dejaba lápida, así que el `status()` siguiente decía «consumido»
  y el usuario leía *«✔️ ya se usó — la acción se hizo con el primer
  toque»* sobre algo que nunca corrió. Apareció al escribir los tests del
  store nuevo, y estaba también en el de memoria. Arreglado en ambos: el
  pop de un vencido es no-op y el status sigue diciendo «vencido».
- **El prompt es sensible al ORDEN, no solo al contenido.** Agregar los dos
  ejemplos de `explicar` al principio del bloque rompió el fixture del
  barrido de ENCUT: de 3/3 a 0/3, emitiendo dos pasos donde va uno.
  Moverlos al final lo devolvió a 8/8. El tablero
  (`docs/scoreboard_router.json`) está re-medido.
- **`explicar` es inalcanzable con un pendiente vivo.** `handle_text`
  deriva a `_apply_edit` antes de rutear, así que preguntar mientras el
  bot espera un dato se interpreta como el dato. El caso de la bitácora no
  cae ahí, pero el hueco quedó fijado en un test. Cerrarlo pide que
  `_apply_edit` reconozca una pregunta como hoy reconoce «cancelar».

## Plan

Cuatro fases, ordenadas por lo que le cuesta al usuario. Deliberadamente
**no** incluye tocar el descompositor, la capa de errores ni el merge de
ediciones: eso ya anda, y la batería lo prueba.

### Fase 0 — Cerrar los agujeros de medición (1 día)

1. **Arreglar D4** (una línea: pasar `chat.id`, no `chat`) y agregar el test
   que lo hubiera visto: construir el `CallbackQuery` con los tipos reales de
   `python-telegram-bot` y verificar el tipo de lo que llega al servicio.
   Sin eso, la Fase 1 entrega un aviso de vencimiento que sigue sin salir.
2. **Arreglar el cluster de prueba** (D6, parte entorno): `vasp_std` en el
   `PATH` de los workers o `BECARIO_VASP_PRELUDE` con el `module load`.
   Mientras no esté, ningún escenario de envío prueba nada más allá del
   `sbatch`.
3. **Adoptar la batería como gate**, con `--repeticiones 3`. Con un LLM de
   por medio la métrica es la tasa, no el booleano: «2 de 3» es un dato.

### Fase 1 — Que el bot no se olvide (3–5 días)

Ataca D5. Es lo único de la lista que hoy le hace perder trabajo al usuario
de forma invisible.

1. **`SQLiteConfirmationStore` y `SQLitePendingStore`.** El puerto ya existe
   (`ports.py:258`): es una implementación nueva, no un rediseño. El TTL pasa
   a ser una columna, no la vida del proceso.
2. **Distinguir «venció» de «se perdió».** Con los pendientes en disco, uno
   vencido da el aviso bueno y `HELP_TEXT` vuelve a significar lo que dice.
3. **Barrer al arrancar** los que vencieron durante la caída, y avisar.

**Cómo se verifica:** un escenario nuevo que reinicie el servicio entre dos
turnos — hoy la batería no puede expresarlo, y por eso D5 no aparece en
rojo. Es la extensión más importante del harness.

### Fase 2 — Que el bot escuche lo que pregunta (2–3 días)

Ataca D3 y D2, en ese orden.

1. **Toda repregunta arma pendiente.** Si un handler devuelve un texto que
   termina en pregunta, deja el pedido esperando. Unificar los dos caminos
   de `preparar_calculo` (compuesto y material faltante), que hoy preguntan
   igual y recuerdan distinto.
2. **Una intención para preguntar** (D2): responder sobre el mensaje anterior
   sin tocar el cluster. Empezar por lo barato — «dónde buscaste», «qué
   parámetros vas a usar» — que se contesta con estado que el bot ya tiene.

**Sale de acá:** CV11 y CV22.

### Fase 3 — Modelar el workspace (3–5 días) · ✅ hecha, con dos desvíos

Ataca D1 y D6. Dos puntos se ejecutaron distinto de como estaban escritos,
y conviene que quede el porqué:

- **No se acotaron las rutas absolutas** (punto 2). Lo implementé y lo
  saqué: bloqueaba pedidos legítimos («creá `/data/proyectos/x`» en un
  área compartida) para tapar un síntoma cuya causa era otra. `/Zr/bcc` no
  lo escribió el usuario, lo INVENTÓ el modelo porque el home no era
  expresable — y eso lo cierra `base` en el origen. El aislamiento sigue
  donde dice ADR-0004: en los permisos del cluster. El parámetro
  `solo_workspace` quedó en el código, sin usar, por si los datos dicen
  otra cosa más adelante.
- **No se hizo el chequeo de arranque de `remote_base` compartido**
  (punto 4). Medido en el cluster de prueba: `stat -f` devuelve
  `ext2/ext3` para `/data` (que SÍ es compartido) y `overlayfs` para
  `/root` (que no), en los dos nodos. El tipo de filesystem no los
  distingue, así que la heurística barata daría una falsa alarma en cada
  arranque justo en este entorno. La versión correcta —escribir un testigo
  y leerlo desde un nodo— exige mandar un trabajo en cada arranque:
  lento, con cola de espera impredecible y ensuciando la contabilidad del
  usuario. Mejor candidato: un `--check-cluster` a pedido, no en el
  arranque.

En su lugar se adelantó **D7** (que estaba en Fase 4) porque es barato,
correcto y ataca el mismo problema desde el otro lado: el diagnóstico
ahora lee el código de salida en vez de conjeturar.

1. **Un `RemotePath` del dominio con tres formas**: `home`, relativa al
   workspace, absoluta. Que «mi home» sea expresable sin inventar texto.
2. **Validar las rutas absolutas** contra las raíces permitidas, no solo
   anclar las relativas. Hoy cualquier `/…` pasa sin mirar — así se crearon
   `/Zr/bcc` y compañía en julio. El síntoma no se reprodujo hoy, pero el
   agujero sigue abierto.
3. **Sacar `/home/ana` del prompt.** Ejemplos con marcadores (`<carpeta>`) o
   rutas del workspace real. La instrucción de «no copiar» se borra: deja de
   hacer falta cuando no hay nada que copiar.
4. **Chequeo de arranque de `remote_base` compartido** (D6): escribir un
   archivo testigo y leerlo desde un nodo, o comparar `stat`. Fail-fast con
   mensaje claro, como ya se hace con Ollama.

**Sale de acá:** CV02 y CV03.

### Fase 4 — Deuda medida (cuando haya lugar)

- **D7:** el diagnóstico arranca por `sacct` (`ExitCode`, `DerivedExitCode`,
  `Reason`) y traduce los códigos frecuentes; las hipótesis, si van, van
  marcadas como hipótesis.
- Unificar el flag `awaiting_params` entre el primer y el segundo ✏️
  (nota de CV24). Cosmético, pero es la clase de incoherencia que después
  hace dudar de un test.

---

## Lo que NO hay que tocar

En un plan de refactor todo parece candidato. Estos no lo son, y la batería
es la razón:

- **El descompositor de pedidos compuestos.** CV09 y CV10 pasan: 7 pasos con
  `formula` en todos. Era la falla más cara de la bitácora y ya no está.
- **La capa de errores hacia el usuario.** CV13, CV14 y CV26 pasan: ni ASE ni
  pydantic se filtran más.
- **El merge de ediciones.** CV15 pasa: `fluorita a=5.07` ya no convierte el
  ZrO2 en CaF₂.
- **La repregunta por fase de un compuesto.** Es lo mejor que hace el bot
  —nombra la fase estable, lista alternativas, avisa cuando la elegida no es
  el fundamental— y CV16 a CV20 lo confirman con cuatro variantes de tipeo.
- **`PlanExecutor` sin rollback.** Correcto para operaciones no
  transaccionales (ADR-0006).
- **La arquitectura por capas.** Que persistir las confirmaciones sea
  «escribir una implementación del puerto» y no «rediseñar» es exactamente el
  retorno de haberla hecho bien.

---

## Cómo se mide que salió bien

`replay_conversaciones.py --repeticiones 3`. El objetivo no es 27/27: es que
**ningún escenario quede en 0/3**, y que los que queden en 1/3 o 2/3 sean
inestabilidad del muestreo y no del sistema — se distingue mirando si la
falla es siempre la misma.

| Escenario | Qué prueba | Fase | Hoy |
|---|---|---|---|
| CV02, CV03 | «mi home» es el home, no las corridas | 3 | ❌ |
| CV22 | una pregunta se responde, no se ejecuta | 2 | ❌ |
| CV11 | la repregunta espera la respuesta | 2 | ❌ |
| (nuevo) | el pendiente sobrevive a un reinicio | 1 | no existe |
| (nuevo) | el aviso de vencimiento llega al chat | 0 | no existe |
