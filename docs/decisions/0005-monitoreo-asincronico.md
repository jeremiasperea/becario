# ADR-0005: Monitoreo asincrónico de trabajos (cierre del loop)

## Contexto

Hasta esta iteración, el bot era puramente reactivo: enviaba un cálculo y
ahí terminaba su responsabilidad — para saber si terminó, el usuario tenía
que volver a preguntar "¿cómo va el 4242?". El objetivo declarado del
proyecto es que BECARIO pueda encadenar pasos de forma autónoma
(estructura → envío → resultado); eso es imposible sin que el sistema
sepa, sin que se lo pregunten, cuándo un trabajo terminó.

## Decisión

Un `JobMonitorService` nuevo, en la capa de aplicación, corre
periódicamente (`poll_and_notify`) sobre los trabajos activos
(`JobTracker.active_jobs()`), consulta su estado real vía
`ClusterGateway.job_state()` — con la conexión SSH propia de cada
dueño, igual que el resto del sistema — y genera `Notification`s neutras
(chat_id + texto) cuando un trabajo llega a un estado terminal. La
capa de presentación (`TelegramBot`) programa este polling con el
`job_queue` de python-telegram-bot (backed por APScheduler) y es la única
responsable de efectivamente mandar el mensaje.

Un trabajo se empieza a rastrear automáticamente al confirmarse un envío
exitoso: `SSHClusterGateway.submit_job` parsea el ID que devuelve `sbatch`
("Submitted batch job 4242") y lo agrega a `CommandResult.job_id`;
`BecarioService.confirm()` usa ese valor para llamar a `JobTracker.track()`.
Si el usuario cancela el trabajo él mismo, se marca como ya notificado
(`mark_notified`) para que el monitor no le mande un aviso redundante
cuando vea el estado CANCELLED más tarde.

Cuando un trabajo termina, además de notificar se agrega una fila al
historial (`HistoryRepository.add`) — cerrando también un vacío de la
versión anterior, donde nada escribía en esa tabla en producción.

## Consecuencias

**A favor:** el estado "terminal" y su interpretación (`JobStatus.from_slurm`)
vive en el dominio, no en el gateway SSH ni en el monitor — decidir qué
cuenta como "terminado" es política de negocio, testeada aparte de la
infraestructura que la alimenta. El servicio de monitoreo es puro
(sin Telegram), así que se testea con fakes en `tests/test_job_monitor.py`
sin red ni SSH.

**En contra:** el polling tiene latencia — el aviso llega hasta
`BECARIO_MONITOR_INTERVAL` segundos después de que el trabajo realmente
terminó (60s por defecto), no al instante. Una alternativa más inmediata
sería que el propio script del cálculo hiciera un curl/webhook al bot al
terminar, pero eso requeriría modificar cada script de cálculo del grupo
y confiar en que el proceso remoto complete esa notificación incluso si
el nodo de cómputo tiene problemas — el polling desde el lado del bot es
más simple y no depende de nada del lado del cluster.

**Nota de robustez:** si `job_state()` no puede determinar el estado (SSH
caído, `sacct` sin salida), el trabajo se deja como está y se reintenta en
la próxima vuelta — nunca se marca como notificado sin haber confirmado
un estado terminal real.
