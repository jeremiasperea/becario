# ADR-0004: Multiusuario con cuentas SSH propias, sin rol admin

## Contexto

El bot pasó de ser una herramienta personal a una pensada para un grupo de
investigación completo. La primera versión tenía una allowlist plana de
`chat_id` de Telegram: cualquier persona autorizada podía, en principio,
cancelar el trabajo de cualquier otra, y toda conexión al cluster corría
con una única cuenta de servicio compartida — sin trazabilidad real a
nivel del sistema operativo del cluster sobre quién hizo qué.

Cada integrante del grupo ya tiene su propia cuenta de usuario en el
cluster (decisión operativa preexistente, no de este proyecto). El grupo
no necesita un rol tipo PI/admin que vea o cancele trabajos ajenos desde
el bot.

## Decisión

Modelar la identidad como `ClusterIdentity`: `telegram_user_id` ↔
`ssh_user` + `ssh_key_path` propios. Un `UserRegistry` (roster en JSON,
administrado fuera de Telegram con `scripts/manage_users.py` — no existe
comando de alta/baja en el chat) resuelve esa identidad al recibir cada
mensaje. Un `ClusterGatewayFactory` cachea una conexión SSH por cuenta,
nunca por chat: dos personas nunca comparten transporte.

La consecuencia central de este diseño es que **el aislamiento entre
usuarios lo garantiza Slurm/el sistema operativo del cluster, no la
aplicación**: como cada quien opera con su propia cuenta, un intento de
`scancel` sobre el trabajo de otra persona falla por permisos del lado
del cluster, sin necesidad de que `BecarioService` lo prevenga por lógica
propia. La aplicación sí agrega una capa adicional donde el costo de
fallar es bajo y el riesgo de confusión (no de exploit) es real: una
`PendingAction` de confirmación guarda quién la pidió (`requester_id`) y
solo esa persona puede confirmarla o rechazarla, incluso si el bot llegara
a usarse alguna vez en un chat compartido.

El historial de cálculos (`historial_calculos`, SQLite) sí es un recurso
de aplicación, no del sistema operativo, así que ahí la aplicación agrega
su propio filtro: `HistoryFilter.owner_id` se completa siempre con el
remitente autenticado, nunca con un valor que pueda sugerir el LLM —
evita que alguien le pida al router "mostrame el historial de fulano" y
el modelo, sin mala intención pero sin restricción, arme un filtro que
cruce esa frontera.

## Consecuencias

**A favor:** no hay lógica de autorización de "quién puede tocar el
trabajo de quién" para mantener en la aplicación — es una clase entera de
bugs de seguridad que no puede existir porque la responsabilidad vive en
la capa correcta (el cluster). El modelo es simple de auditar: un archivo
JSON dice quién es quién.

**En contra:** si el grupo más adelante quisiera un rol de supervisión
(un PI viendo el estado agregado de todos los trabajos, por ejemplo), hoy
no hay forma de hacerlo sin acceso directo al cluster o una extensión
deliberada de `UserRegistry` con un campo de rol — se dejó fuera de
alcance a propósito en esta iteración, ver la respuesta correspondiente
en la sesión de diseño.

**Alternativa descartada:** una cuenta de servicio compartida con
`sudo`/`sacct --allusers` habría requerido que la aplicación implementara
su propio control de acceso sobre "trabajos ajenos", duplicando lo que
Slurm ya resuelve, con más superficie para errores.
