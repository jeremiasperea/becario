# Plan de tolerancia a fallos

Este plan sí salió de leer el código — al revés que `plan_refactor_bot.md`, que
salió de la bitácora. Conviene decirlo de entrada, porque cambia cuánto hay que
creerle a cada parte: los defectos están verificados en el código con archivo y
línea, pero **que hayan causado un fallo en producción se verificó solo en tres
casos**, y están marcados como tales. El resto son agujeros abiertos, no
incidentes.

Aun así, la bitácora tenía la respuesta guardada. De los 118 mensajes de
`chat_messages`, **tres pedidos del usuario no recibieron ninguna respuesta**, y
el log de decisiones del router dice qué pasó con dos de ellos.

## El diagnóstico en una línea

El bot tolera bien los fallos que **previó**, y no tolera que falle el canal.

Eso no es un detalle de vocabulario. `CommandResult(ok=False)`, el
`Optional[bool]` de `file_exists` que distingue «no está» de «no pude mirar», el
fail-closed de Materials Project, los dos TTL separados: todo eso está pensado y
está bien. Es manejo de errores, y es de los buenos.

Tolerancia a fallos es otra cosa: qué hace el sistema cuando se rompe algo que
nadie modeló. Hoy, tres respuestas posibles — se cuelga, se calla, o pierde el
trabajo — y ninguna se lo dice a nadie.

## La evidencia de la bitácora

### Dos respuestas que se evaporaron

```
[ 23] 2026-07-18T05:16:01 user: 'mostrame la estructura de archivos en el cluster'
[ 24] 2026-07-18T05:50:45 user: 'dentro de Zr quiero crees 3 carpetas bcc, fcc y hcp'
```

Treinta y cuatro minutos de silencio, y el usuario se rindió y pidió otra cosa.
Pero el router **sí funcionó**: `decisiones_router` #10, 3.4 s,
`outcome='routed'`, `[{"action": "listar_archivos"}]`. Ruteó bien, el handler
corrió, el servicio devolvió un `Reply` con `ok=True` — y al chat no llegó nada.

```
[ 87] 2026-08-02T03:43:48  bot: '📊 Estado trabajos (root): Error de conexión SSH…'
[ 88] 2026-08-02T21:29:04 user: 'relajá el ZrO2'
[ 89] 2026-08-02T21:39:59 user: 'relajá el ZrO2'
```

`decisiones_router` #39: 108.5 s de latencia, `outcome='error'`, registrado a las
21:30:52 — o sea **108 segundos después del mensaje 88, exactamente**. El
servicio terminó y marcó la decisión. El usuario esperó **diez minutos y
cincuenta y cinco segundos** y repitió.

Las dos cuentan la misma historia: `handle_text` devolvió un `Reply`, y el
mensaje murió en el camino a Telegram. Es la firma de **T1** — una excepción
después de que el servicio devuelve, sin nadie que la atrape.

No puedo decir **qué** excepción, y no voy a inventarlo: el mensaje 23 es de
julio y el código de entonces no es el de hoy. Lo que sí se puede afirmar es
dónde estaba el agujero, porque sigue abierto.

### El router vive pegado a su propio timeout

`decisiones_router`, 44 decisiones:

| | latencia |
|---|---|
| mínimo | 3.4 s |
| mediana | 30.3 s |
| p90 | **118.1 s** |
| máximo | 267.9 s |

`BECARIO_OLLAMA_TIMEOUT` está en **120 s**. El p90 real es 118.1. Catorce de las
44 decisiones pasaron los 100 s.

El sistema opera, de forma rutinaria, a dos segundos de su propio límite. No hace
falta que Ollama se caiga: alcanza con que la máquina esté un poco más cargada
que de costumbre para que el pedido expire — y cuando expira, `_chat` devuelve
`None`, `route()` lo colapsa en `Intent.UNKNOWN`, y el usuario lee *«No pude
interpretar tu pedido»*. Ese texto aparece **tres veces** en la bitácora.

De las 44 decisiones, 17 quedaron en `outcome='error'`. El 39 %.

---

## Los defectos

### T1 — Una excepción no prevista deja al usuario en silencio

No existe `add_error_handler` en todo el repositorio (0 coincidencias).
`_on_text` (`telegram_bot.py:158`) no tiene `try`/`except`. Cualquier excepción
—un `sqlite3.OperationalError`, un `KeyError` en un payload, un reventón de
pymatgen, un `BadRequest` de Telegram al enviar— sube a python-telegram-bot, se
loguea, y el usuario **no recibe nada**. Se le corta hasta el «escribiendo…».

Es el defecto de los mensajes 23 y 88.

Y tiene un agravante propio. En `confirm` (`services.py:1035`) el token se
consume con `pop` **antes** de ejecutar:

```python
plan = self._confirmations.pop(token)  # recién ahora se consume
...
_ok, text = executor(ctx, action)      # si esto revienta, el plan ya no existe
```

Si la ejecución lanza una excepción, el plan se evaporó y el usuario **no sabe si
el `sbatch` salió o no**. En un bot que dispara trabajos a un cluster, esa
ambigüedad es lo más caro de toda la lista.

### T2 — El SSH puede colgarse indefinidamente

`ssh_gateway.py:114` espera el resultado con `recv_exit_status()`. En el paramiko
instalado (`channel.py:400`) eso es:

```python
self.status_event.wait()   # sin timeout
```

El `timeout=` que recibe `exec_command` es el timeout de lectura del canal, no
del código de salida — lo dice su propio docstring: *«set command's channel
timeout»*.

¿Y por qué no se despierta solo? Porque **no hay `set_keepalive` en ningún
lado** (0 coincidencias). Ante un corte silencioso —VPN, wifi, una NAT que
expira sin mandar RST— el TCP nunca se entera, paramiko nunca detecta el cierre,
y el hilo queda bloqueado hasta que alguien reinicie el proceso.

De yapa, `_connection()` (`ssh_gateway.py:88`) decide si reusar la conexión con
`transport.is_active()`, que sobre un socket medio abierto sigue contestando
`True`.

### T3 — El monitor corre SSH bloqueante dentro del event loop

`telegram_bot.py:281`:

```python
for note in self._job_monitor.poll_and_notify():
```

Llamada síncrona, con N trabajos × varias idas y vueltas SSH cada uno, cada 60
segundos, **directamente en el loop de asyncio**. `_on_text` sí usa
`_run_blocking` y lo manda a un hilo; el monitor se quedó afuera.

Los tres primeros se componen, y esa es la parte grave: **con T2, un solo SSH
colgado durante un tick del monitor congela el bot entero, para todos los
usuarios, de forma permanente.** No se pone lento. No vuelve.

### T4 — El aviso de fin de trabajo se marca entregado antes de entregarlo

`job_monitor.py:103` ejecuta `mark_notified`, y recién después
`telegram_bot.py:281-293` intenta el `send_message`. Si el envío falla, el
`except` loguea el error y sigue de largo — pero el trabajo ya salió de
`active_jobs()`.

El aviso se perdió **para siempre**. Es entrega *at-most-once* sobre lo único que
cierra el loop de todo el sistema (ADR-0005).

La «nota de robustez» del ADR-0005 dice, con razón, que nunca se marca como
notificado sin confirmar un estado terminal real. El agujero es el otro: se marca
sin confirmar que el **aviso** salió.

### T5 — Cero reintentos en los tres bordes de red

Una llamada, sin backoff, en los tres:

- **Ollama** (`ollama_router.py:541`): `_chat` devuelve `None` y `route()` lo
  convierte en `Intent.UNKNOWN`. Un fallo de infraestructura disfrazado de fallo
  de comprensión, que manda al usuario a reescribir un mensaje que estaba
  perfecto. Con el p90 pegado al timeout (arriba), esto no es hipotético.
- **SSH** (`ssh_gateway.py:120`): devuelve `ok=False` al primer error.
- **Materials Project** (`materials_project.py:58`): sin timeout propio y sin
  reintento.

### T6 — Trabajos zombi en el tracker

`active_jobs()` (`storage.py:742`) filtra solo por `notified = 0`. Cuando Slurm
purga un trabajo viejo por `MinJobAge`, `job_state` devuelve `None` y
`job_monitor.py:77` anota *«reintento la próxima vuelta»*… para siempre. No hay
contador de intentos, ni antigüedad máxima, ni un estado «lo perdí de vista».

Cada trabajo perdido cuesta una consulta SSH cada 60 segundos, indefinidamente.

### T7 — Estado remoto huérfano

`calc.py:591` sube el directorio de la corrida **antes** de pedir la
confirmación. Si el usuario cancela, deja vencer el TTL, o el bot se reinicia,
los inputs quedan tirados en el cluster sin nadie que los limpie.

Y `upload_dir` (`ssh_gateway.py:243`) no es atómica: un fallo a mitad de camino
deja un directorio a medio subir que igual «existe» para cualquiera que lo mire.

### T8 — Sin cierre ordenado

`close_all()` está escrito (`ssh_gateway.py:368`) y **nunca se llama**: la única
coincidencia en el repositorio es su propia definición. No hay handler de
`SIGTERM`. Un `systemctl restart` deja los transportes abiertos hasta que el
sistema operativo los recicle.

### T9 — El límite de Telegram se controla por handler, no en el borde

`remote_files.py:21` define `_LISTING_MAX_CHARS = 3500` y lo aplica en dos
lugares: el listado (`:196`) y el contenido de un archivo (`:267`). Bien.

El problema es dónde está puesto. El límite de 4096 caracteres es de **Telegram**
—vive en `_send_reply`, `telegram_bot.py:133`— y ahí no lo mira nadie. Todo lo
que no pase por esos dos handlers viaja sin red: el reporte de un plan de siete
pasos, el diagnóstico de fallo con dos colas de log (`job_monitor.py:130`), la
tabla del barrido de ENCUT. Esa última encima va envuelta en `html.escape`, que
**agranda** el texto.

Un guard por handler protege los casos que alguien se acordó de proteger. El
borde protege todos.

### T10 — Observabilidad

No hay id de correlación entre lo que ve el usuario y lo que quedó en el log. Los
`logger.error` no distinguen «el cluster dijo que no» de «no hay cluster». Nadie
cuenta cuántas veces falló el LLM. No hay health check.

Para un servicio que corre desatendido, el único síntoma es alguien diciendo
«che, no me contesta» — que es, textualmente, lo que pasó el 18 de julio y el 2
de agosto.

---

## Estado de ejecución

Al 2026-08-11. La suite pasó de **972 a 1050 tests**, todo verde. **Las cuatro
etapas están hechas**; queda pendiente T10 (observabilidad), que nunca tuvo
etapa propia.

| Etapa | Estado | Qué se hizo |
|---|---|---|
| — | ✅ | `BECARIO_OLLAMA_TIMEOUT` de 120 a 180 s |
| 0 | ✅ | T1 (error handler + confirmación blindada) y T9 (troceado en el borde) |
| 1 | 🔶 | T2 (keepalive + deadline) y T3 (monitor a un hilo). Un desvío, abajo |
| 2 | ✅ | T5 completo: mensaje honesto, tercer estado y reintentos |
| 3 | ✅ | T4 (acuse de entrega) y T6 (racha de consultas sin respuesta) |
| 4 | ✅ | T7 (`.pending/` + `mv` + barrido) y T8 (cierre ordenado) |

### Cómo quedó T7

Los inputs se suben a `{base}/.pending/{run_name}` y se mueven a su lugar
definitivo con un `mv` recién al confirmar. El `mv` dentro del mismo filesystem
es atómico, así que la corrida aparece entera o no aparece — antes, una subida
cortada a la mitad dejaba un directorio que existía, parecía válido y le
faltaba el POTCAR.

El `.pending/` no es solo prolijidad: hace que un directorio abandonado sea
**reconocible**. Uno suelto en la base de corridas es indistinguible de una
corrida real, y nadie se anima a borrar lo que no puede identificar. Por eso
antes no se limpiaba nada.

Tres caminos de limpieza, en orden de precisión:

1. **Cancelar (❌)** borra la corrida en el acto.
2. **El barrido por edad** (>2 h), que corre al preparar un cálculo nuevo sobre
   la cuenta de esa persona. Cubre lo que el borrado explícito no puede: nadie
   aprieta ❌ cuando deja **vencer** una confirmación, y un reinicio del bot
   entre la subida y el botón no deja a nadie a quien avisarle.
3. Nada más. No hace falta: el estado está en el propio cluster, así que barrer
   por edad no exige llevar registro de nada.

El `rm -rf` es la única operación destructiva del gateway y va con su propio
guard además del `shlex.quote`: exige ruta absoluta, sin `..`, dentro de un
`.pending/` y que no sea el `.pending/` mismo. Está ejercitado con seis rutas
que **no** debe aceptar, incluidas `/`, una corrida real y
`/data/runs/.pending/../../../etc`.

**Hallazgo aparte:** `ConfirmationStore.purge_expired()` está definido en el
puerto y en las dos implementaciones, y **no lo llama nadie**. No se cableó
acá porque el barrido por edad ya cubre el problema remoto sin depender de él;
queda anotado como deuda, porque las filas vencidas se acumulan en SQLite.

### Lo que destrabó el tercer estado

`CommandResult.reason` (`COMMAND` / `TRANSPORT` / `TIMEOUT`) y su propiedad
`transitorio` no son un refinamiento cosmético: eran la precondición de otras
dos cosas.

1. **Los reintentos.** Sin la distinción, cualquier reintento reintentaba
   también los `permission denied`, que es latencia y ruido de log a cambio de
   nada.
2. **El umbral de T6 bajó de 60 a 5.** `job_state()` ahora devuelve
   `JobStateReading(state, reachable)` en vez de un `Optional[str]` que juntaba
   «`sacct` contestó y no lo conoce» con «no pude preguntar». El monitor cuenta
   la racha **solo** con respuestas explícitas del cluster, así que cinco ya son
   concluyentes — antes hacían falta sesenta para no soltar un trabajo sano por
   un rato de red mala.

### Qué se reintenta y qué no

Esta es la parte peligrosa de la etapa, y por eso el default del gateway es
**no** reintentar: cada operación que sí lo hace lo declara en su llamada.

| Operación | Reintenta | Por qué |
|---|---|---|
| `sbatch` | **no** | No es idempotente. Si el envío llega y la respuesta se pierde, reintentar encola el trabajo dos veces: el usuario no ve el primero (no tenemos su id) y paga las horas igual |
| `scancel` | sí | Cancelar algo ya cancelado no hace nada |
| `sacct`, `squeue`, `tree`, `find`, `$HOME` | sí | Solo leen |
| `mkdir -p` | sí | Idempotente por definición |
| Ollama, salvo TIMEOUT | sí | Falla en milisegundos y se arregla solo seguido |
| Ollama con TIMEOUT | **no** | Ya se esperaron 180 s; repetir le suma otros 180 al usuario para llegar al mismo lugar |
| Materials Project, solo `NETWORK` | sí | Consultar no tiene efectos; un `NO_MATCH` no cambia por insistir |

Hay un test por cada fila peligrosa, y el del `sbatch` está verificado contra el
cambio que lo rompería: marcarlo reintentable lo hace fallar (1 intento contra
3).

### Desvío en la Etapa 3: la antigüedad no sirve como criterio

El punto 2 pedía soltar un trabajo «después de N consultas sin respuesta **o X
horas**». Se implementó solo lo primero, y la segunda mitad se descartó: un
trabajo puede estar legítimamente encolado durante días, así que la antigüedad
no distingue «perdido» de «esperando turno». Soltar por viejo habría dejado de
avisar sobre trabajos sanos que simplemente tardaban.

Tampoco se agregó `last_seen_at`: con la racha en `poll_attempts` no hace falta
para decidir nada, y una columna que no decide nada es una que se desactualiza
en silencio.

El umbral quedó **holgado a propósito** (60 consultas seguidas, una hora con el
intervalo por defecto). El motivo es que hoy `job_state()` devuelve el mismo
`None` para «`sacct` no lo conoce» y para «el SSH está caído». Hasta que el
gateway distinga esos dos casos —es la parte de la Etapa 2 que falta— conviene
errar del lado de seguir esperando: soltar un trabajo sano porque el cluster
estuvo caído veinte minutos sería peor que el problema que se está arreglando.

### Desvío en la Etapa 1: no se puso un `wait_for` por pedido

El punto 4 de la etapa pedía «un `wait_for` global por pedido, así un hilo
colgado no se lleva la capacidad entera». **Se hizo el pool acotado y propio
(`becario-blocking`, 8 hilos) pero NO el `wait_for`,** y conviene que quede el
porqué.

Un `wait_for` no cancela un hilo que ya está corriendo: la corrutina se
abandona, el hilo sigue. O sea que no libera capacidad — que era el argumento
para ponerlo. Y sí introduce el daño que la Etapa 0 acababa de sacar: el usuario
recibiría «tardó demasiado» mientras el `sbatch` sigue camino al cluster, que es
exactamente la ambigüedad que `_fallo_tras_confirmar` existe para evitar.

Quien libera el hilo de verdad es el deadline del gateway, porque corta abajo,
donde el bloqueo ocurre. Ese es el arreglo; el `wait_for` habría sido su
apariencia.

**El mensaje honesto se adelantó** (punto 3 de la Etapa 2) porque el resto de
esa etapa —partir `ok=False` en dos, backoff— es bastante más grande, y este
pedazo solo se paga a sí mismo: el timeout de 180 s compra margen, pero cuando
venza igual, el usuario tiene que enterarse de que expiró y no de que se
explicó mal.

`RouterUnavailableError` + `RouterFailureReason` (TIMEOUT / UNREACHABLE / API)
viven en el dominio con el mismo patrón que `StructureResolutionError`. Son tres
motivos y no uno porque llevan a **acciones** distintas: ante un timeout tiene
sentido reintentar; ante un servidor caído hay que avisarle a alguien.

Cinco cosas que salieron de ejecutarlo y no estaban en el diagnóstico:

- **El arreglo de `confirm` era el opuesto del que estaba escrito.** Está
  explicado arriba, en el punto 2 de la Etapa 0. La lección para el resto del
  plan: antes de reordenar algo que parece un descuido, mirar si no es un guard.
- **El troceado no puede medir con `len()`.** La ruta monospace envuelve el texto
  en `<pre>{html.escape(...)}</pre>`, y `html.escape` **agranda**: 3000 `&`
  crudos son 15000 caracteres enviados. Medir el texto crudo dejaba pasar
  exactamente los mensajes que el límite tenía que atajar. `_trocear` recibe la
  función de medida, y el corte de una línea larga se busca por bisección
  porque la medida no es lineal.
- **Cada trozo tiene que ser HTML válido solo.** Un `<pre>` que abre en un
  mensaje y cierra en el siguiente no lo renderiza nadie. Está fijado en un test.
- **Hacer que el router levante excepción rompía los harness de medición.**
  `_majority_check` (`live_router_check.py`) no atrapaba nada: antes un timeout
  volvía como plan `UNKNOWN` y contaba como fixture errado, y con la excepción
  mataba la corrida entera — que son ~2 horas. Igual en los tres brazos de
  `medir_schemas_router.py`. Quedó mejor que antes: el intento sigue contando
  como fallido, pero ahora el reporte dice `router no disponible (timeout)` en
  vez de confundirlo con una respuesta mala del modelo.
- **`extract_structure` es la única llamada que se traga el fallo**, y tiene que
  serlo: es una segunda pasada de mejora sobre un plan que `route()` ya
  devolvió. Hacer fallar el pedido entero por la pasada opcional cambiaría una
  respuesta parcial por ninguna.
- **Escribí un test de concurrencia decorativo y casi me lo creo.** El primero
  de T3 usaba `time.sleep()` en el monitor y medía cuánto tardaba en atender un
  mensaje *después* de arrancar el tick. Daba verde — pero también daba verde
  con el bug puesto, porque cuando el monitor corre inline el bloqueo pasa
  **mientras el test todavía espera que el tick arranque**: para cuando uno mira
  el reloj, ya pasó. Un test de concurrencia con `sleep` mide el reloj del test,
  no el del sistema. La versión que quedó usa dos `threading.Event` (el monitor
  avisa que entró y espera permiso para salir), y está verificada contra el
  comportamiento viejo: **3.00 s contra un umbral de 1.50 s.** Un test así hay
  que correrlo con el bug puesto, o no se sabe si prueba algo.
- **Un doble con forma propia deja de avisar cuando el tipo real crece.** Los
  tests del tick del monitor armaban la notificación con `SimpleNamespace`, así
  que al agregarle `acuse` al `Notification` verdadero el doble siguió andando y
  el camino nuevo habría quedado sin cubrir. Ahora usan el dataclass real. Es la
  misma familia que D4: el doble se parecía lo suficiente para pasar y no lo
  suficiente para servir.
- **El backoff hizo dormir a la suite.** Al agregar reintentos en los tres
  bordes, los tests que ejercitan un fallo transitorio empezaron a pagar las
  pausas de verdad: +3 s de suite, y creciendo con cada test nuevo. La costura
  está en `tests/conftest.py`, pero el primer intento fue apagar `time.sleep` a
  secas y rompió siete tests que lo usan legítimamente para hacer vencer un
  TTL. Por eso `reintentos.dormir_backoff` existe como función aparte: es lo
  que permite apagar exactamente el backoff y nada más.

Queda pendiente, y no lo pude tocar: **`.env.example`**, si documenta el
timeout viejo (el archivo está fuera de mi alcance de permisos).

### La batería contra el sistema vivo

Corrida con las cuatro etapas puestas, contra el cluster de prueba, Ollama y
Materials Project reales: primero **27/28 en 3/3** (2816 s), y **28/28 en 3/3**
(3439 s) una vez arreglado el harness — ver abajo. `qwen2.5-coder:14b`.

El único en rojo fue **CV11, y no era del bot**. La misma corrida sobre el
código **anterior** a todo este trabajo da idéntico: 1/5, con el mismo mensaje.
O sea que ninguna de las cuatro etapas metió una regresión en el sistema
completo — que es lo que la batería existe para contestar, porque todo lo demás
que se verificó son dobles que fallan a propósito.

Lo que sí apareció es **un defecto del harness**, y de los caros: hacía que un
escenario sano pareciera inestable.

`run_scenario` decía en su docstring que los pendientes y las confirmaciones
"viven en memoria por usuario", y por eso construía un servicio nuevo por
escenario. Era cierto cuando se escribió. Dejó de serlo cuando se persistieron
en SQLite (Fase 1 de `plan_refactor_bot.md`): desde entonces dos servicios
comparten el archivo y construir uno nuevo no limpia nada.

El síntoma solo salía con `--repeticiones`. CV11 **termina** con una repregunta
abierta (pregunta la fase del ZrO2), así que ese pendiente quedaba vivo y la
repetición siguiente interpretaba su primer mensaje —«relajá el bulk»— como la
respuesta a la pregunta anterior, en vez de rutearlo de cero. El runner creía
estar aislando por `chat_id` («chat_id propio por corrida») pero los pendientes
se indexan por `user_id`, que era siempre el mismo.

Resultado: **CV11 daba 1/1 y 1/5.** Esa diferencia es exactamente la que
`--repeticiones` existe para detectar —«está roto» contra «sale una de cada
tres»— y acá la estaba fabricando el instrumento. Con el pendiente limpiado al
arrancar cada corrida: **5/5**, y CV28 (que prueba que el estado SÍ sobrevive a
un reinicio) sigue verde. La batería completa re-corrida quedó en **28/28**.

La lección se parece a D4 y no es casualidad: **un comentario que explica por
qué algo alcanza deja de ser verdad en silencio cuando cambia lo que
describe.** Acá el que cambió fue un store que se volvió persistente a
propósito, y el harness siguió confiando en una garantía que ya no tenía.

## Plan

Cuatro etapas, ordenadas por lo que le cuesta al usuario cada defecto.

### Etapa 0 — Red de contención (1 día)

Ataca **T1**. Es chico, es casi todo en presentación, y es lo único de la lista
que ya se sabe que le costó respuestas a un usuario real.

1. **`add_error_handler`** que loguee con traza y le conteste al usuario: «se me
   rompió algo procesando esto, **no ejecuté nada**», con un id de correlación
   que aparezca también en el log.
2. **Envolver la ejecución** de un plan ya confirmado, dejando el `pop` donde
   está.

   > **Corrección.** La primera versión de este plan decía «`peek` → ejecutar →
   > `pop`», y estaba mal. `pop` es el guard atómico
   > (`UPDATE … WHERE consumed_at IS NULL`, `storage.py:540-546`): es lo que hace
   > que dos toques simultáneos de ✅ no manden **dos `sbatch`**. Con `peek`
   > primero, los dos toques pasan el chequeo y los dos ejecutan. Habría
   > arreglado T1 rompiendo una garantía que este código construyó a propósito.

   Y el token **no se repone** tras el fallo, aunque tiente: la excepción pudo
   saltar *después* de que el `sbatch` saliera, así que reponerlo es ofrecerle al
   usuario un botón que duplica un trabajo ya encolado. Lo único honesto es
   decir que no sabemos y mandar a **verificar**, no a reintentar.
3. **Trocear en `_send_reply`** (T9): partir en mensajes de ≤4000 caracteres en
   el borde, midiendo **después** del `html.escape`. Los guards de
   `remote_files.py` se pueden quedar donde están; dejan de ser la única defensa.

**Cómo se verifica:** un test de frontera que haga fallar el `send_message` y
verifique que el usuario recibe *algo*. Es la misma clase de test que faltaba en
D4 — *la suite prueba la función, nadie prueba la llamada*.

### Etapa 1 — Que nada bloquee para siempre (2–3 días)

Ataca **T2** y **T3**. Es la de mayor impacto por lejos: convierte un cuelgue
permanente en un error que se recupera solo.

1. **`transport.set_keepalive(30)`** en `_connection`. Una línea, y es la que
   hace que paramiko se entere de que la conexión murió.
2. **Esperar el exit status con deadline**: `channel.status_event.wait(timeout)`
   y, si vence, cerrar el canal y devolver un `CommandResult` de timeout.
3. **`poll_and_notify()` a `asyncio.to_thread`**, igual que `handle_text`.
4. **Executor propio y acotado** para el trabajo bloqueante, con un `wait_for`
   por pedido. Hoy `asyncio.to_thread` usa el executor default y compartido: un
   hilo colgado se lleva capacidad de todos.

**Cómo se verifica:** un `ClusterGateway` fake que se cuelga, y un test que
exige que el bot siga contestando a otro usuario mientras tanto.

### Etapa 2 — Distinguir lo transitorio de lo definitivo (3–4 días)

Ataca **T5**. El punto 1 es la precondición de todo lo demás.

1. **Un tercer estado en `CommandResult`.** Hoy `ok=False` mezcla «mkdir dijo
   permission denied» con «no hay red». El primero no se reintenta jamás; el
   segundo sí. Sin esa distinción, cualquier reintento que agregues reintenta las
   dos cosas — y reintentar un `permission denied` es puro ruido.
2. **Backoff exponencial con jitter** en los tres bordes, con un solo helper
   compartido. No uno por adaptador.
3. **Que `IntentRouter.route` pueda señalar el fallo** con una excepción tipada,
   como ya hace `ensure_model_available` (`ollama_router.py:504`), en vez de
   colapsarlo todo en `UNKNOWN`. Ahí el mensaje pasa a ser honesto: «no puedo
   pensar ahora, Ollama no responde» en vez de «no pude interpretar tu pedido».
4. **Revisar `BECARIO_OLLAMA_TIMEOUT`** a la luz del p90 de 118 s. Subirlo es lo
   fácil; lo correcto es que el punto 3 exista primero, para que expirar deje de
   ser indistinguible de no entender.

### Etapa 3 — Entrega confiable del cierre del loop (2–3 días)

Ataca **T4** y **T6**.

1. **Invertir el orden**: enviar, y recién ahí `mark_notified`. Cambio chico de
   interfaz — el bot llama a `tracker.mark_notified` después del `send_message`
   exitoso, o el monitor recibe un acuse.
2. **Columnas `poll_attempts` y `last_seen_at`** en `trabajos_monitoreados`.
   Después de N consultas sin respuesta o X horas, marcar el trabajo como perdido,
   avisar **una sola vez** («perdí el rastro del trabajo 4242») y soltarlo.

### Etapa 4 — Consistencia del estado remoto · ✅ hecha

Ataca **T7** y **T8**.

1. **Subir a `.pending/` y renombrar al confirmar** (el `mv` remoto es atómico).
   La alternativa —subir recién en `confirm`— es más simple pero alarga la espera
   entre el botón y el «listo», que es justo el momento en que el usuario está
   mirando.
2. **Borrar el `run_dir`** cuando vence o se cancela una confirmación.
3. **Handler de `SIGTERM`** que llame a `close_all()`, que ya existe.

---

## Lo que NO hay que tocar

- **`PlanExecutor` sin rollback.** Es correcto para operaciones no
  transaccionales, y ADR-0006 ya explica por qué. Nada de esta lista lo cambia.
- **El fail-fast del arranque** (`main.py:250-253`). Validar el token y el modelo
  antes de arrancar es lo contrario de un problema de tolerancia a fallos: es
  fallar temprano y con un mensaje claro, que es exactamente lo que corresponde
  cuando el fallo es de configuración y no se va a arreglar reintentando.
- **El aislamiento por cuenta SSH** (ADR-0004). Una conexión por identidad es más
  frágil que un pool compartido, y así tiene que ser.
- **La arquitectura por capas.** Que «reintentar SSH» sea envolver un adaptador y
  no rediseñar tres módulos es, otra vez, el retorno de haberla hecho bien.

---

## Cómo se mide que salió bien

La batería de conversaciones no sirve para esto: corre en un proceso sano contra
un cluster que anda. Prueba que el sistema funciona, no que aguanta.

Lo que falta es una batería de **fallos inyectados**, y hoy no existe ninguna:

Vive en `tests/test_tolerancia_fallos.py`:

| Prueba | Qué exige | Ataca | |
|---|---|---|---|
| una excepción no atrapada | el usuario recibe un aviso, no silencio | T1 | ✅ |
| el aviso del incidente también falla | no hay bucle; queda en el log | T1 | ✅ |
| el executor revienta tras confirmar | manda a verificar, no a reintentar | T1 | ✅ |
| doble ✅ con la ejecución rota | el token no se repone: un solo `sbatch` | T1 | ✅ |
| un `Reply` de 8000 caracteres | llegan varios mensajes, no cero | T9 | ✅ |
| 3000 `&` en una tabla monospace | el escape cuenta para el límite | T9 | ✅ |
| el modelo expira | dice «tiempo de espera», no «no te entendí» | T5 | ✅ |
| los tres motivos de fallo | dan tres mensajes distintos | T5 | ✅ |
| el modelo falla mientras se edita | el plan sigue en el estante | T5 | ✅ |
| httpx timeout / connect / 500 | se traducen al motivo correcto | T5 | ✅ |
| un comando SSH que nunca termina | corta por deadline y cierra el canal | T2 | ✅ |
| la conexión SSH se abre | pide keepalive al transporte | T2 | ✅ |
| el monitor trabado adentro del cluster | el bot atiende igual, en otro hilo | T3 | ✅ |
| `send_message` falla al notificar el fin | no se acusa: el trabajo sigue activo | T4 | ✅ |
| el aviso sale bien | recién ahí se asienta el trabajo | T4 | ✅ |
| la cosecha del barrido (2º mensaje) | no se acusa: no duplica el historial | T4 | ✅ |
| `sacct` no conoce más el trabajo | se suelta tras la racha, con aviso | T6 | ✅ |
| el cluster vuelve a contestar | la racha se reinicia | T6 | ✅ |
| una base sin la columna nueva | se migra sin perder los trabajos | T6 | ✅ |
| **un `sbatch` que falla por red** | **NO se reintenta: no se duplica** | T5 | ✅ |
| `scancel`, `mkdir`, `sacct` por red | sí se reintentan | T5 | ✅ |
| Ollama expira | no se reintenta (no sumarle 180 s al usuario) | T5 | ✅ |
| Ollama inalcanzable | se reintenta 3 veces | T5 | ✅ |
| el cluster caído durante el poll | no suma a la racha del trabajo | T5, T6 | ✅ |
| el backoff | crece, respeta el tope y reparte con jitter | T5 | ✅ |
| seis rutas que no son de pendientes | el `rm -rf` se niega, sin ejecutar nada | T7 | ✅ |
| confirmar | mueve de `.pending/` al lugar definitivo | T7 | ✅ |
| el `mv` falla | no se envía nada al cluster | T7 | ✅ |
| cancelar | borra la corrida subida | T7 | ✅ |
| el borrado falla al cancelar | la cancelación se completa igual | T7 | ✅ |
| preparar un cálculo | barre los pendientes viejos de esa cuenta | T7 | ✅ |
| el bot termina (o revienta) | se cierran las conexiones SSH | T8 | ✅ |

Ninguna necesita cluster, ni Ollama, ni red — todas son fakes que fallan a
propósito. Esa es la razón de que puedan existir, y también de que no
existieran: **las fallas que cubren no se reproducen pidiéndole cosas a un
sistema sano.** Sin ellas, cada arreglo de este plan se vuelve a romper en tres
meses y nadie se entera, que es precisamente lo que este documento trata de
explicar.
