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
python scripts/manage_users.py add --telegram-id 111111111 \
    --ssh-user jperez --ssh-key /home/becario/.ssh/id_jperez --name "Juan Pérez"
python scripts/manage_users.py list
python scripts/manage_users.py remove --telegram-id 111111111
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
python main.py

# 2) Registrá tu cuenta del cluster (interactivo: te pregunta los datos):
python scripts/manage_users.py add

# 3) Volvé a arrancar; ya no vuelve a pedir nada.
python main.py
```

Los secretos y datos locales (`.env`, `users.json`, `becario.db`) están en
`.gitignore` y **no** deben subirse a ningún repositorio. El `.env` se crea con
permisos `600` (solo tu usuario puede leerlo).

Las variables disponibles (todas opcionales salvo `BECARIO_BOT_TOKEN` y
`BECARIO_SSH_HOST`) están documentadas en `.env.example`. Si preferís, podés
setearlas como variables de entorno: el entorno tiene prioridad sobre el `.env`.

> Si el bot venía de usar webhooks: `curl https://api.telegram.org/bot<TOKEN>/deleteWebhook`

### Como servicio (systemd)

```ini
# /etc/systemd/system/becario.service
[Unit]
Description=BECARIO Telegram HPC bot
After=network-online.target

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

## Tests

```bash
python -m pytest            # 144 tests
python -m pytest --cov=becario --cov-report=term-missing
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
