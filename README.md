# B.E.C.A.R.I.O.

Asistente de Telegram para clusters HPC con Slurm: lenguaje natural → intención (LLM local) → acciones validadas sobre el cluster vía SSH.

## Arquitectura

Clean Architecture con dependencias apuntando hacia el dominio:

```
presentation/telegram_bot.py     ← Telegram (python-telegram-bot, long polling)
        │
application/services.py          ← Casos de uso (BecarioService)
        │  depende solo de ↓
domain/models.py + ports.py      ← Entidades validadas (Pydantic) + interfaces (Protocol)
        ▲  implementados por
application/
    services.py                  ← BecarioService (casos de uso conversacionales)
    job_monitor.py                ← JobMonitorService (polling + notificación, cierra el loop)
infrastructure/
    ollama_router.py             ← IntentRouter    (Ollama structured outputs)
    ase_builder.py               ← StructureBuilder (ASE: bulk/moléculas/POSCAR)
    ssh_gateway.py               ← ClusterGateway + ClusterGatewayFactory
                                   (paramiko + shlex.quote + SFTP, una conexión por cuenta)
    user_registry.py             ← UserRegistry (roster JSON, sin altas por Telegram)
    storage.py                   ← HistoryRepository, JobTracker (SQLite, por owner_id)
                                   ConfirmationStore (memoria + TTL, ownership por token)
main.py                          ← Composition root (única DI del proyecto)
scripts/manage_users.py          ← CLI para administrar el roster del grupo
```

Decisiones de diseño grandes documentadas como ADRs en `docs/decisions/`
(pensadas para citar/adaptar en el capítulo de arquitectura del informe).

### SOLID aplicado

- **S**: cada clase tiene una responsabilidad (el bot traduce updates, el servicio orquesta, el gateway ejecuta).
- **O**: para agregar una intención nueva se agrega un handler al dict de `BecarioService.handle_text` y un método al gateway — sin tocar lo existente.
- **L**: los fakes de los tests sustituyen a las implementaciones reales sin que el servicio lo note.
- **I**: puertos chicos y específicos (`IntentRouter`, `ClusterGateway`, `HistoryRepository`, `ConfirmationStore`, `Transcriber`).
- **D**: la aplicación depende de `Protocol`s del dominio; paramiko/Ollama/SQLite son detalles inyectados en `main.py`.

### Multiusuario (ver ADR-0004)

No hay rol admin dentro del bot. Cada integrante del grupo tiene su propia
cuenta SSH en el cluster; el bot resuelve `telegram_user_id → ClusterIdentity`
contra un roster (`UserRegistry`) y abre/cachea una conexión SSH **por
cuenta**, nunca compartida. El aislamiento entre personas lo hace cumplir
el propio cluster: `scancel`/`sacct` sobre un trabajo ajeno falla por
permisos del sistema, no por lógica de la aplicación.

Administrar el roster (dar de alta/baja a alguien) es una operación fuera
de Telegram, con `scripts/manage_users.py`:

```bash
python3 scripts/manage_users.py add --telegram-id 111111111 \
    --ssh-user jperez --ssh-key /home/becario/.ssh/id_jperez --name "Juan Pérez"
python3 scripts/manage_users.py list
python3 scripts/manage_users.py remove --telegram-id 111111111
```

### Seguridad (defensa en profundidad)

1. **Identidad por persona**, no allowlist plana: sin entrada en el roster, el bot responde y no ejecuta nada (`NOT_REGISTERED_TEXT`).
2. **Validación semántica** en el dominio: `SlurmJobRequest`, `JobId`, `HistoryFilter`, `ClusterIdentity` con regexes ancladas en `\A...\Z` (¡no `$`, que permite `\n` final!).
3. **Quoting sintáctico** en el gateway: todo argumento pasa por `shlex.quote`; el script sbatch viaja por heredoc quoteado.
4. **SQL siempre parametrizado** (placeholders `?`), y el historial se filtra por `owner_id` asignado por el servicio — nunca por un valor que el LLM pueda sugerir.
5. **Confirmación humana** con botones inline para `sbatch` y `scancel`; los tokens se consumen una sola vez, expiran (TTL 10 min) y **solo puede confirmarlos quien las pidió** (`requester_id`), aunque el bot se usara en un chat compartido.

## Setup

Cada persona corre su propia copia en su máquina, con su token de bot y su
cuenta personal del cluster. No hace falta exportar variables a mano: la
primera vez que arrancás, un **asistente interactivo** te pide los datos y los
guarda en un archivo `.env` **local** (que no se sube a git, ver `.gitignore`).

```bash
pip install -e ".[dev]"

# 1) Arrancá: la primera vez, el asistente te pide token del bot, host del
#    cluster y Ollama, y crea el .env por vos.
python3 main.py

# 2) Registrá tu cuenta del cluster (interactivo: te pregunta los datos):
python3 scripts/manage_users.py add

# 3) Volvé a arrancar; ya no vuelve a pedir nada.
python3 main.py
```

Los secretos y datos locales (`.env`, `users.json`, `becario.db`) están en
`.gitignore` y **no** deben subirse a ningún repositorio. El `.env` se crea con
permisos `600` (solo tu usuario puede leerlo).

Las variables disponibles (todas opcionales salvo `BECARIO_BOT_TOKEN` y
`BECARIO_SSH_HOST`) están documentadas en `.env.example`. Si preferís, podés
setearlas como variables de entorno: el entorno tiene prioridad sobre el `.env`.

> Si el bot venía de usar webhooks: `curl https://api.telegram.org/bot<TOKEN>/deleteWebhook`

### Arranque con Ollama automático

`main.py` **no** levanta Ollama: valida que esté arriba con el modelo configurado
y aborta con un mensaje claro si no (fail-fast, ver ADR-1/ADR-2). Levantar un
demonio del sistema es orquestación, no responsabilidad del composition root.

Para eso hay un launcher una capa más afuera:

```bash
python3 scripts/start_becario.py        # asegura Ollama y arranca el bot
python3 scripts/start_becario.py --yes  # sin preguntas (systemd, CI)
python3 scripts/start_becario.py --check-only   # deja Ollama listo, no arranca el bot
```

Qué hace, en orden:

1. Si no existe el comando `ollama`, te muestra el comando del instalador
   oficial y pregunta si lo instala.
2. Consulta `BECARIO_OLLAMA_URL`. Si no responde y es **local**, corre
   `ollama serve` en background (log en `ollama.log`, gitignoreado) y espera
   hasta `--timeout` segundos a que atienda.
3. Si falta el modelo de `BECARIO_OLLAMA_MODEL`, consulta su **peso** en el
   registro de Ollama, te muestra el **espacio libre** en el store de modelos y
   recién ahí pregunta si lo baja:

   ```
   ⚠️  El modelo 'qwen2.5:7b' no está instalado en Ollama.
      Peso de la descarga: 4.7 GB
      Espacio libre en /home/vos/.ollama/models: 123.4 GB
      ¿Lo bajo ahora con `ollama pull qwen2.5:7b`? [s/N]
   ```

   Si el modelo **no entra en disco**, corta sin bajar nada. Ese chequeo no lo
   saltea `--yes`: esa flag aprueba la intención de bajar, no hace aparecer
   espacio. Si el peso no se puede consultar (registro caído, sin red), avisa y
   sigue igual — es un dato informativo, no un requisito.
4. Le cede el proceso a `main.py` con `execv`, así el bot queda con un solo PID
   y las señales (Ctrl+C, `systemctl stop`) le llegan directo.

`--yes` aprueba todo sin preguntar (instalación incluida): es para systemd y CI,
donde no hay nadie mirando. Sin terminal, cualquier pregunta cuenta como "no".

Si la URL apunta a **otra máquina**, el launcher no intenta nada: avisa qué
falta allá y corta. No hay forma de arrancar un demonio remoto desde acá.

El servidor que levanta queda corriendo después de que el bot termine (es un
demonio); el propio script te imprime el PID para pararlo.

> **Ojo con el store de modelos.** `ollama serve` resuelve los modelos contra el
> home del usuario que lo corre. Un Ollama de sistema (usuario `ollama`) usa
> `/usr/share/ollama/.ollama` y **no ve** los modelos que bajaste con tu usuario
> en `~/.ollama`. Si `ollama list` te muestra modelos pero el bot dice que falta
> el modelo, es esto: bajalo con `sudo -u ollama ollama pull <modelo>` o pará el
> servicio de sistema y dejá que el launcher levante uno con tu usuario.

### Como servicio (systemd)

```ini
# /etc/systemd/system/becario.service
[Unit]
Description=BECARIO Telegram HPC bot
After=network-online.target ollama.service
Wants=ollama.service

[Service]
User=jeremias
WorkingDirectory=/opt/becario
EnvironmentFile=/opt/becario/.env
ExecStart=/opt/becario/.venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Con systemd de por medio, la dependencia declarativa (`Wants=`/`After=`) es
mejor que el launcher: el init system ya sabe ordenar y reiniciar servicios, y
`main.py` mantiene su fail-fast. `start_becario.py` es para la máquina de
escritorio, donde no hay nadie ordenando el arranque.

## Tests

```bash
python3 -m pytest            # 154 tests
python3 -m pytest --cov=becario --cov-report=term-missing
```

La suite cubre: sanitización contra inyección de shell y SQL, parseo
defensivo de la salida del LLM, flujo de confirmación (ejecuta exactamente
una vez, tokens idempotentes, con expiración y con verificación de que
solo quien la pidió puede confirmarla), aislamiento multiusuario (dos
personas nunca comparten conexión SSH ni ven el historial de la otra),
construcción de comandos SSH (sin conexión real, interceptando `_run`) y
generación de estructuras con ASE real (bulk, moléculas G2, superceldas,
POSCAR válidos releídos con `ase.io.read`).

## Estructuras atómicas (ASE)

`modificar_estructura` construye la estructura **localmente** con ASE y la
sube al cluster por SFTP si se indica destino. Ejemplos de pedidos:

- "Generá un POSCAR de silicio 2x2x2"
- "Armá NaCl rocksalt con a=5.64 y subilo a /home/user/calculos"
- "Creá una molécula de H2O con 12 Å de vacío en formato xyz"

Soporta: bulk de elementos (estructura de referencia de ASE) y compuestos
(indicando red + parámetro), moléculas de la base G2, superceldas hasta
10×10×10, vacío, y salida VASP/CIF/XYZ.

## Cálculos VASP completos (`preparar_calculo`)

Además de estructuras sueltas, el bot sabe preparar y lanzar un cálculo
VASP completo: genera **localmente** POSCAR + INCAR + KPOINTS + script de
corrida, sube todo por SFTP, arma el POTCAR concatenando los de la
biblioteca del cluster (`BECARIO_POTCAR_DIR`, con búsqueda de variantes
`X_sv` → `X_pv` → `X`), y deja el `sbatch` esperando tu confirmación.
Ejemplos de pedidos:

- "Relajá los parámetros de red del bulk de W" → relajación con `ISIF=3`
- "Energía estática de Zr hcp con ENCUT 450"
- "Hacé la curva de convergencia de ENCUT para Zr hcp de 250 a 450"

El barrido de ENCUT corre como **un solo job secuencial** (un
subdirectorio `encut_NNN/` por punto). Cuando el monitor detecta que
terminó, **cosecha las energías** de los `OSZICAR` remotos y te manda la
tabla E(ENCUT) con el ENCUT recomendado (menor valor a <1 meV/átomo del
máximo del barrido):

```
📈 Convergencia de ENCUT (Zr_convergencia_encut):

 ENCUT          E (eV)   ΔE (meV/át)
------------------------------------
   250      -16.893412         28.29
   300      -16.948113          0.94
   ...
✅ ENCUT recomendado: 300 eV (ΔE < 1 meV/át respecto de 450 eV)
```

La confirmación ofrece tres botones: **✅ Confirmar**, **❌ Cancelar** y
**✏️ Modificar** — este último espera un mensaje con el cambio ("usá 2
nodos", "subí el ENCUT máximo a 600"); el resto del plan se mantiene y se
rearma la confirmación, iterando las veces que haga falta. Si un cambio
no se entiende, el plan sigue vivo y se vuelve a preguntar: solo un
«cancelar» explícito lo descarta. Además, cada envío queda registrado con
una huella de sus parámetros: si pedís algo idéntico o muy similar a una
corrida previa, la confirmación te lo avisa (con fecha, job y directorio)
y podés correrlo igual o usarlo de base con ✏️ Modificar.

Los resultados de corridas previas se consultan en lenguaje natural
("dame los parámetros de red del zirconio"): el bot busca tu corrida más
reciente de ese material (prefiere relajaciones), lee el CONTCAR remoto y
responde a, b, c, ángulos y energía final.

Requiere configurar en el `.env`: `BECARIO_POTCAR_DIR` (biblioteca en el
cluster), `BECARIO_VASP_CMD` y opcionalmente `BECARIO_VASP_PRELUDE` y
`BECARIO_REMOTE_BASE` (ver `.env.example`).

## Cierre del loop: seguimiento de trabajos (ver ADR-0005)

Enviar un cálculo activa su seguimiento automáticamente. Un monitor de
fondo (`JobMonitorService`, programado con el `job_queue` de
python-telegram-bot) revisa cada `BECARIO_MONITOR_INTERVAL` segundos
(60 por defecto) los trabajos activos, y avisa por Telegram apenas
detecta un estado terminal (COMPLETED/FAILED/CANCELLED/TIMEOUT):

```
✅ Tu trabajo 4242 (grafeno_dft) terminó: completed
```

Cada trabajo terminado queda además registrado en el historial, así que
"consultá el historial" empieza a tener datos reales con el uso.

## Pendientes conocidos

- `Transcriber` es un puerto sin implementación incluida: enchufá tu
  servicio Whisper implementando `transcribe(audio_bytes) -> str` y
  pasándolo a `TelegramBot`.
- `ConfirmationStore` es en memoria: las confirmaciones pendientes se
  pierden al reiniciar el proceso (comportamiento seguro; si querés
  persistencia, implementá el puerto sobre SQLite).
- El `IntentRouter` usa *structured outputs* de Ollama: el JSON Schema se
  deriva de `RouterDecision` (Pydantic) y el modelo queda obligado a
  responder conforme al schema. Es la alternativa correcta al tool calling
  para modelos Gemma, que no exponen esa capacidad en Ollama.
- El "modo consultar y sugerir" (el LLM propone un plan sin ejecutar nada,
  usando historial + estado como contexto) todavía no está implementado —
  es el siguiente paso natural ahora que hay datos de historial reales
  para que el LLM pueda razonar sobre ellos.
