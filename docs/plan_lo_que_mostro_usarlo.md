# Lo que mostró usarlo

Los dos planes anteriores salieron de leer: `plan_refactor_bot.md` de la
bitácora vieja, `plan_tolerancia_fallos.md` del código. Este salió de
**usar el bot**, el 2026-08-20, en la primera sesión real después de
levantarlo con el entorno de VASP completo.

Es la fuente más cara de las tres y la más honesta: no hay que inferir qué
haría el usuario, porque el usuario está ahí escribiendo.

## Los números

Veinte turnos: cinco botones y **quince pedidos en lenguaje natural**. De
esos veinte, **diez recibieron una respuesta problemática** — un error, un
aviso, o algo que directamente no era lo pedido. La mitad.

Lo que funcionó bien conviene decirlo primero, porque acota el trabajo:

| Pedido | Resultado |
|---|---|
| «¿qué me falta para el Zr?» | ✅ la sugerencia nueva, correcta |
| «relajá el bulk de Zr hcp» | ✅ preparó, confirmó, envió, **completó** |
| «Calcula el bulk de W bcc» | ✅ ídem, job 15 completado |
| «mostrame el CONTCAR» | ✅ resuelto contra la última corrida |
| «Muéstrame el OSZICAR» | ✅ ídem |

El camino central —pedir un cálculo, confirmarlo, que corra, leer un
archivo del resultado— **anda de punta a punta**. Los seis defectos de
abajo están alrededor de ese camino, no adentro.

---

## E1 — «el último cálculo» no es expresable, y el modelo lo inventa

Cuatro intentos de pedir el listado de archivos de una corrida. Ninguno
funcionó. Lo que el router emitió, textual:

```
listar_archivos(base=corridas, destino_remoto=<nombre_del_ultimo_calculo>)
listar_archivos(base=corridas, destino_remoto=run_14)
```

La primera línea es un **placeholder literal**, con corchetes y todo. La
segunda es un directorio **inventado**: `run_14` no existe en ningún lado.

No es que el modelo no entendiera. Entendió perfecto y no tuvo con qué
decirlo: `base` conoce `home`, `corridas` y `absoluta`, y no existe forma
de anclar a «la corrida más reciente» ni a «la corrida del job N» — aunque
el bot **tiene** ese dato, en `corridas_vasp.run_dir`.

**Esto es D1 otra vez.** Aquella vez el concepto inexpresable era «mi
home» y el modelo inventaba `/home/ana`, que además era un ejemplo del
prompt. Se arregló agregando el enum `base`. La lección que quedó escrita
—*«un campo del router por cada feature no escala»*, ADR-0006— sigue
valiendo, pero acá el problema no es un campo más: es que **la misma
capacidad existe en el handler de al lado**.

`ver_archivo` resuelve un nombre suelto contra el directorio de la última
corrida. Su docstring dice, textual: *«esto último arregla el pedido real
"ver el CONTCAR", que antes el LLM resolvía inventando un `ls`»*. Por eso
«mostrame el CONTCAR» funcionó y «mostrame los archivos» no. Es la prueba
de que el defecto es del schema y no del modelo.

Y hay un agravante: la palabra «cálculo» tira fuerte hacia el historial.
Tres de los cuatro intentos rutearon a `consultar_db + listar_archivos`, y
el usuario se quedó mirando una tabla de historial que no había pedido.

## E2 — una ruta que no existe se ve como un directorio vacío

```
vos> lista de archivo en /data/becario_runs/Zr_relajacion
bot> 📂 /data/becario_runs/Zr_relajacion:
     /data/becario_runs/Zr_relajacion
```

Ese directorio **no existe**. Las corridas llevan timestamp
(`Zr_relajacion_20260819_225757`). El bot afirma que existe y está vacío.

Reconstruida, la cadena es:

1. `tree` no está instalado en el cluster → `tree: command not found`.
2. `list_directory` cae al fallback de `find`, que es correcto.
3. `find <ruta inexistente> | sort` devuelve **`ok=True` con stdout vacío**:
   el error va a stderr y el exit code se pierde en el pipe.
4. `_format_tree` recibe vacío y dibuja solo la raíz.

De los seis defectos **este es el único que miente**. Los demás dejan al
usuario sin respuesta —molesto, pero honesto—; este da una respuesta falsa
con cara de correcta, y sobre el pedido más común que hay. Es la misma
familia que el tag del INCAR mal escrito que VASP ignora, o el CONTCAR de
una relajación que agotó el NSW: silenciosamente equivocado.

## E3 — un callback vencido se reporta como si el bot se hubiera roto

```
vos> cancelar
bot> ⚠️ Se me rompió algo procesando tu pedido y no ejecuté nada.
     […] incidente a3141a51
```

Dos veces seguidas (a3141a51 y 6bfa0651). El traceback:

```
telegram.error.BadRequest: Query is too old and response timeout expired
  telegram_bot.py:426 → await query.answer()
```

La tarjeta tenía **quince minutos y cuarenta y seis segundos**. Telegram
caduca los callbacks, y `query.answer()` sobre uno vencido levanta
`BadRequest`.

Lo notable es que **la línea siguiente ya contempla exactamente esta
condición**:

```python
await query.answer()                                 # ← 426, revienta
try:
    await query.edit_message_reply_markup(reply_markup=None)
except Exception:  # p. ej. mensaje demasiado viejo para editar
```

Quien escribió eso previó el mensaje viejo para el `edit` y no para el
`answer`, una línea antes.

El costo real: la confirmación **ya había vencido** por su propio TTL (600 s),
así que `reject` iba a contestar *«⌛ Esta confirmación expiró o ya fue
usada»* — un mensaje útil, que explica qué pasó. En su lugar el usuario
recibió un incidente, que suena a que el bot está roto cuando lo único que
pasó es que se tardó.

**Y hay un riesgo latente peor.** El servicio ejecuta la acción en la línea
403; `query.answer()` revienta en la 426, **después**. Acá no pasó nada
porque el token ya estaba vencido, pero con un ✅ sobre una confirmación
todavía viva, el `sbatch` sale y el usuario lee *«no ejecuté nada»*. La
frase sería falsa, y es justo la ambigüedad que la Etapa 0 existía para
cerrar.

## E4 — un pedido imposible produce un plan plausible y sin sentido

El más grave de todos.

```
vos> Arma una supercelda de Zr sobre W 2x2/3x3 respectivamente,
     no olvides agregas 15 ang de vacío en z
```

El bot contestó con un batch de ocho pasos, prolijo y numerado, listo para
aprobar de un botón. Lo que ruteó:

```
3. modificar_estructura(formula=Zr, red_cristalina=bcc, supercelda=[2,2,2])
4. modificar_estructura(formula=W,  red_cristalina=fcc, supercelda=[2,2,2])
6. modificar_estructura(formula=Zr, supercelda=[1,1,1], encut_min=250, encut_max=450)
8. modificar_estructura(formula=Zr_on_W, red_cristalina=bcc_fcc, …)
```

Todo lo que está mal ahí:

| | |
|---|---|
| **Física** | W como `fcc` — es **bcc**. Zr como `bcc` — es **hcp** |
| **Parámetros** | pidió 2×2 y 3×3; emitió `[2,2,2]` en los dos |
| **Perdido** | los 15 Å de vacío en z **no aparecen en ningún paso** |
| **Inventado** | `formula=Zr_on_W` no es una fórmula; `red_cristalina=bcc_fcc` no es una red |
| **Contaminación** | `encut_min/encut_max` son de un *cálculo*, no de una estructura |
| **Redundancia** | los pasos 2 y 5 crean la misma carpeta |

Pero la raíz no es ninguna de esas: **el bot no sabe apilar dos materiales.**
Una heterostructura Zr-sobre-W no está en `StructureKind` ni en ningún
lado. El pedido era imposible desde el principio, y en vez de decirlo se
inventó una interpretación y la presentó con ocho pasos y un botón.

Peor: el preview del batch **valida**, pero no lo suficiente.
`validate_structure_params` chequea que haya fórmula y que una losa traiga
su cara, y no construye el `StructureRequest`. Si lo construyera, el paso 8
reventaría en el acto — `StructureRequest(formula='Zr_on_W')` **rechaza**
esa fórmula. O sea que el batch le ofreció al usuario aprobar un paso que
está garantizado que falla.

Y un hallazgo lateral: `elements_of('Zr_on_W')` devuelve `['Zr']`. No
rechaza: **descarta en silencio** el resto de la cadena. Así es como el
disparate sobrevive hasta el preview.

## E5 — un plan de ocho pasos, y ninguna forma de corregirlo

```
vos> Aquí debería haber una opción de modificar
bot> 🛠️ Así estoy configurado: • POTCAR: los busco en /data/potcars…
```

Dos cosas de una. La primera, que tenías razón: frente al batch más
equivocado de la sesión, las únicas salidas eran ✅ y ❌. `allow_modify`
existe y se usa en el camino de un solo cálculo; el preview del batch no lo
ofrece.

La segunda, que tu observación se ruteó a `explicar` y te devolvió el
volcado de configuración. Es D2 un escalón más arriba: el bot sabe
contestar «¿dónde buscaste?», pero no reconoce un comentario sobre **su
propia interfaz**.

## E6 — preguntas sobre el contenido de un archivo que ya mostró

```
vos> Cuantas vueltas iónicas hizo?
bot> ❓ No pude interpretar tu pedido.
```

Ruteó a `ver_archivo(OSZICAR) + explicar()` y terminó en el texto de ayuda.
El bot **acababa de mostrar** ese OSZICAR: la respuesta estaba en su propio
mensaje anterior. Misma familia que D2 y que E5 — la diferencia es que acá
no alcanza con leer el archivo, hay que interpretarlo (contar las líneas de
paso iónico).

---

## Qué tienen en común

Cuatro de los seis (E1, E2, E4, E6) son **la misma forma de fallar**: el
bot no distingue «no puedo» de «puedo». Frente a algo que no sabe hacer no
se planta; produce lo más parecido que encuentra y lo presenta con la misma
confianza que una respuesta buena.

Eso está bien documentado para el LLM —por eso existen las repreguntas, el
fail-closed de Materials Project, el vocabulario de tags— pero **el patrón
se rompe en los bordes nuevos**: el listado de archivos, el batch de
estructuras, las preguntas sobre contenido.

## Plan

Ordenado por lo que le cuesta al usuario, no por dificultad.

### 1. Que nada mienta (E2, E4-preview)

- `list_directory` tiene que distinguir «vacío» de «no existe». El `find`
  del fallback pierde el exit code en el pipe; hay que mirarlo, y si la
  ruta no existe decirlo — mejor aún, ofrecer lo que sí hay en el
  directorio padre, como ya hace el «¿quisiste decir…?» de los tags.
- `validate_structure_params` tiene que construir el `StructureRequest`
  en vez de chequear un subconjunto a mano. Es puro y sin I/O: es
  exactamente lo que el preview necesita, y hoy deja pasar pasos que
  fallan seguro.
- `elements_of` no puede descartar en silencio: si la fórmula no parsea
  entera, es un error, no un `['Zr']`.

### 2. Que el bot diga «no sé hacer eso» (E4)

Una heterostructura no está en `StructureKind`. El pedido tiene que
rebotar con un mensaje claro, no con ocho pasos inventados. Es la misma
decisión que se tomó con el barrido de k-points en las sugerencias:
**no ofrecer lo que no se sabe hacer.**

### 3. Que un callback viejo no parezca un crash (E3)

Envolver `query.answer()` igual que la línea de abajo, y —esto importa
más— **mover la ejecución después del acuse**, o dejar constancia de que
ya se ejecutó, para que el mensaje de incidente nunca pueda decir «no
ejecuté nada» sobre algo que sí se hizo.

### 4. Que «el último cálculo» sea expresable (E1)

La capacidad ya existe en `ver_archivo`. Dársela a `listar_archivos` es
darle a `base` una forma más de anclar —la corrida más reciente, o la del
job N— en vez de esperar que el modelo escriba una ruta que no puede
conocer. Ojo con el presupuesto de schema y con el orden de los ejemplos
en el prompt: los dos ya mordieron antes.

### 5. Que el batch se pueda corregir (E5)

`allow_modify` en el preview del batch. Y evaluar si un comentario sobre
la interfaz —«acá debería haber…»— merece su propio camino o si alcanza
con que `explicar` sepa hablar del mensaje anterior.

### 6. Preguntas sobre contenido (E6)

El más grande de los seis y el menos urgente. Requiere que el bot
interprete un archivo, no que lo muestre. Conviene medirlo antes de
diseñarlo, como se hizo con el router en dos etapas.

#### Lo que dijo la medición

Se midió antes de diseñar, con `scripts/medir_preguntas_de_contenido.py`
sobre nueve preguntas —la real más ocho de la misma familia, marcadas como
sintéticas— y el resultado corrige el párrafo de arriba. **Estas preguntas
no piden interpretación: piden un parseo, y la mayoría de esos parseos ya
están escritos.**

*Brazo A — inventario, sin LLM. **6 de 9 ya se calculan en el repo.***

La pregunta real es una de ellas:

```
✅ [REAL ] Cuantas vueltas iónicas hizo?
       ya está en: relaxed_source._IONIC_STEP_RE (se usa en _check_convergence)
       verificado sobre el fragmento real -> 2
```

`_check_convergence` cuenta los pasos iónicos en cada relajación, para
avisar si se quedó sin NSW. El bot **tenía el número**, lo dice en otra
frase por otro motivo, y cuando se lo preguntaron contestó «no pude
interpretar tu pedido». Es E1 un escalón más adentro: la capacidad está en
el módulo de al lado y no hay cómo nombrarla.

De las nueve, una sola pide interpretación de verdad —«resumime qué dice el
OUTCAR», que no tiene un hecho puntual que calcular—. Las otras dos que
faltan son parsers chicos: leer un tag del INCAR, contar pasos electrónicos.

*Brazo B — ruteo, con Ollama. **3 planes distintos para 9 hechos
distintos; 7 de 9 preguntas comparten plan con otra.***

El router no se pierde: manda siete de las nueve a `consultar_resultados`,
3/3 unánime. El problema es que ese handler contesta UNA cosa fija —los
parámetros de red y el E0 de la última corrida— sin importar qué se
preguntó. Dos preguntas caen bien de casualidad (justo piden eso); las
otras cinco reciben, con toda confianza, la respuesta a otra pregunta.

O sea que el ruteo **pierde el pedido**: «cuántas vueltas iónicas hizo» y
«qué energía dio» producen el mismo plan byte a byte. No falta un handler
que interprete archivos — falta que el plan pueda decir QUÉ hecho se pidió.

Un hallazgo lateral: el OSZICAR que el bot mostró estaba **truncado**
(`… (archivo truncado)`), así que ni el humano podía contar sobre lo que
vio. La pregunta no era comodidad, era la única salida.

*Brazo C — el vocabulario candidato. **18/18 elige bien, 9/9 se
abstiene**, 3/3 unánime en las nueve.*

El brazo C le da al plan un campo para nombrar el dato, con un enum de seis
valores: exactamente los seis que el brazo A encontró ya calculados. Lo que
no está adentro tiene que contestar `ninguno` — el punto 2 de este mismo
plan, *no ofrecer lo que no se sabe hacer*, ahora medible: tres de las nueve
preguntas están afuera a propósito.

Por eso la métrica va partida y no promediada. Un brazo que acierta todo lo
que conoce y nunca se abstiene sería E4 con otra cara: a «resumime el
OUTCAR» le contestaría la energía, con toda confianza. Las tres
abstenciones son la mitad que más valía medir, y salieron 9/9.

De paso: 2.5 s por llamada contra los ~10 s del schema grande.

**La conclusión es la misma que con el enum `base`: el problema era que
faltaba cómo decirlo.** No hace falta un camino que interprete archivos.
Hacen falta un campo y seis parsers que ya están escritos.

#### Lo que esta medición NO dice

Tres límites, para que nadie la lea de más:

1. El corpus tiene **una** pregunta real y ocho que escribí yo, y el enum
   salió de la misma lista. Que alineen no es evidencia fuerte: está
   medido contra sí mismo. Las abstenciones son lo único que podría haber
   salido mal y no salió.
2. El brazo C es una llamada AISLADA con schema chico. En producción el
   campo va al schema grande —donde muerden el presupuesto de ADR-0006 y
   el orden de los ejemplos, las dos cosas que este plan ya vio romper— o
   a una segunda pasada tipo `extract_structure`. Cuál de las dos es la
   decisión que sigue, y se mide antes de elegirla.
3. Nada de esto midió las RESPUESTAS, solo el ruteo. Que el bot sepa que le
   pidieron `pasos_ionicos` no es que sepa contestarlo bien sobre un
   OSZICAR de 2 MB.

Ojo con dos cosas al leer el reporte: el corpus tiene **una** pregunta real
y ocho inventadas, y la primera versión del brazo B medía «ruteos que
nombran el archivo» y daba 24/27 — un número que sonaba bien y no medía
nada. Está anotado en el script para que no se repita.

## Cómo se verifica

Los seis salen de mensajes reales, así que van a `tests/conversaciones/`
como escenarios nuevos. Tres se pueden fijar con tests unitarios sin red
—E2 (find sobre ruta inexistente), E3 (callback vencido), E4-preview
(`validate_structure_params` con `Zr_on_W`)— y esos son los que más
valen: son los que hoy fallan en silencio.
