"""Adaptador SSH hacia el cluster (implementa ClusterGateway).

Regla de oro: cada argumento que viaja al shell remoto pasa por
`shlex.quote`. Los métodos reciben modelos de dominio ya validados,
así que hay defensa en profundidad: validación semántica (Pydantic)
+ quoting sintáctico (shlex).
"""
from __future__ import annotations

import difflib
import logging
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Optional

import paramiko

from ..domain.models import (
    DIR_PENDIENTES,
    ClusterIdentity,
    CommandFailureReason,
    CommandResult,
    JobId,
    JobStateReading,
    SlurmJobRequest,
)
from ..reintentos import reintentar

logger = logging.getLogger(__name__)

_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def _partes_sanas(path: str) -> Optional[tuple[str, ...]]:
    """Partes de una ruta remota que se puede borrar, o None si no.

    Exige absoluta y sin `..`: lo primero para que no dependa del cwd de la
    sesión, lo segundo porque `.pending/../..` cumpliría cualquier chequeo
    que se limite a buscar el segmento.
    """
    if not path.startswith("/"):
        return None
    partes = PurePosixPath(path).parts
    return None if ".." in partes else partes


def _es_ruta_de_pendientes(path: str) -> bool:
    """¿`path` es UNA corrida sin confirmar (no el contenedor)?"""
    partes = _partes_sanas(path)
    if partes is None or DIR_PENDIENTES not in partes:
        return False
    # Tiene que haber algo DESPUÉS del `.pending/`: borrar el contenedor
    # entero se llevaría por delante los pendientes de otras corridas.
    return partes.index(DIR_PENDIENTES) < len(partes) - 1


def _es_base_de_pendientes(path: str) -> bool:
    """¿`path` es exactamente el contenedor `.../.pending`?"""
    partes = _partes_sanas(path)
    return bool(partes) and partes[-1] == DIR_PENDIENTES


def _format_tree(base: str, find_output: str) -> str:
    """Emula la salida de `tree -L 2` a partir de `find -maxdepth 2`.

    Recibe rutas absolutas (una por línea, en cualquier orden) y dibuja el
    árbol con ramas (├──/└──), igual que `tree`: para decidir la rama de
    cada entrada hace falta saber si es la última hija de su padre, así que
    primero se arma la jerarquía y después se renderiza.

    Los hermanos se ordenan acá, al renderizar, y no con un `| sort` en el
    cluster. Ese pipe se comía el código de salida de `find` —una ruta
    inexistente volvía `ok=True` y sin salida, o sea indistinguible de un
    directorio vacío—, y ordenar por nodo es además más fiel a un árbol que
    ordenar las rutas completas como texto.
    """
    base_depth = len(PurePosixPath(base).parts)
    tree: dict = {}
    for raw in find_output.strip().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = PurePosixPath(raw).parts[base_depth:]
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines = [base]

    def _render(node: dict, prefix: str) -> None:
        names = sorted(node)
        for i, name in enumerate(names):
            last = i == len(names) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}")
            _render(node[name], prefix + ("    " if last else "│   "))

    _render(tree, "")
    return "\n".join(lines)


# Cuántos vecinos se nombran cuando ninguno se parece al pedido: bastantes
# para orientar, pocos para que el mensaje siga entrando en un chat.
_MAX_HERMANOS = 10


def _sugerir_hermanos(nombre: str, hermanos: list[str]) -> str:
    """El «¿quisiste decir…?» de una ruta que no existe, dados sus vecinos.

    Mismo trato que el vocabulario de tags de VASP: decir que algo no existe
    es honesto pero deja al usuario en el mismo lugar. El prefijo va primero
    porque es el caso real —se pide `Zr_relajacion` y la corrida es
    `Zr_relajacion_20260819_225757`— y difflib después, para los tipeos.
    """
    if not hermanos:
        return "El directorio que lo contiene está vacío."
    cerca = [h for h in hermanos if h.startswith(nombre)]
    cerca += [
        h for h in difflib.get_close_matches(nombre, hermanos, n=3, cutoff=0.6)
        if h not in cerca
    ]
    if cerca:
        return "¿Quisiste decir " + " o ".join(f"«{h}»" for h in cerca[:3]) + "?"
    resto = len(hermanos) - _MAX_HERMANOS
    return (
        "Ahí al lado hay: "
        + ", ".join(hermanos[:_MAX_HERMANOS])
        + (f" (y {resto} más)" if resto > 0 else "")
        + "."
    )


class SSHClusterGateway:
    """Gateway al nodo login del cluster mediante paramiko."""

    def __init__(
        self,
        host: str,
        user: str,
        key_path: str,
        port: int = 22,
        connect_timeout: float = 15.0,
        command_timeout: float = 120.0,
        keepalive_interval: float = 30.0,
    ) -> None:
        self._host = host
        self._user = user
        # paramiko no expande '~' en key_filename, y el alta interactiva
        # (`manage_users.py`) sugiere rutas con tilde: se expande acá.
        self._key_path = str(Path(key_path).expanduser())
        self._port = port
        self._connect_timeout = connect_timeout
        self._command_timeout = command_timeout
        self._keepalive_interval = keepalive_interval
        self._client: Optional[paramiko.SSHClient] = None
        self._home_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Conexión (perezosa y reutilizable)
    # ------------------------------------------------------------------
    def _connection(self) -> paramiko.SSHClient:
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=self._host,
            port=self._port,
            username=self._user,
            key_filename=self._key_path,
            timeout=self._connect_timeout,
        )
        # Sin keepalive, una conexión que muere en silencio —VPN que se
        # corta, wifi que cambia, una NAT que expira sin mandar RST— no se
        # entera nunca: el TCP sigue "abierto" para los dos lados, el hilo
        # lector de paramiko se queda esperando bytes que no van a llegar, y
        # cualquier comando en curso cuelga PARA SIEMPRE. Con keepalive el
        # peer muerto se detecta en decenas de segundos, paramiko cierra el
        # transporte, y el `_run` de abajo lo ve como un error normal.
        #
        # Esto arregla la causa; el deadline de `_run` es el paracaídas para
        # todo lo demás que pueda tardar sin límite.
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(int(self._keepalive_interval))
        self._client = client
        return client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _run(self, command: str, *, reintentable: bool = False) -> CommandResult:
        """Corre un comando remoto.

        `reintentable` es **opt-in y por buenas razones**: `sbatch` no es
        idempotente, así que reintentarlo cuando la respuesta se perdió en
        el camino manda el trabajo dos veces y le cobra al usuario dos veces
        las horas de cómputo por un error que nunca vio. El default seguro
        es no reintentar; cada operación que sí puede lo dice en su llamada.
        """
        if not reintentable:
            return self._run_once(command)
        return reintentar(
            lambda: self._run_once(command),
            intentos=3,
            resultado_transitorio=lambda r: r.transitorio,
            etiqueta=f"comando SSH ({command[:40]})",
        )

    def _run_once(self, command: str) -> CommandResult:
        try:
            client = self._connection()
            _, stdout, stderr = client.exec_command(
                command, timeout=self._command_timeout
            )
            channel = stdout.channel
            # `recv_exit_status()` NO tiene timeout: por dentro es un
            # `status_event.wait()` pelado (paramiko `channel.py`), y el
            # `timeout=` de `exec_command` es el de LECTURA del canal, no el
            # del código de salida — lo dice su propio docstring. O sea que
            # esa línea podía esperar para siempre, y el hilo que la llamaba
            # quedaba quemado hasta reiniciar el proceso.
            #
            # `wait(timeout)` devuelve False si venció. El evento también se
            # setea cuando el canal se cierra, así que un canal que muere sin
            # dejar código de salida sale por acá con exit_status = -1, que
            # no es 0 y por lo tanto cuenta como fallo. Correcto: un comando
            # que no pudo decir cómo terminó no terminó bien.
            if not channel.status_event.wait(self._command_timeout):
                channel.close()
                logger.error(
                    "Comando SSH sin respuesta tras %.0f s: %.80s",
                    self._command_timeout, command,
                )
                return CommandResult(
                    ok=False,
                    stderr=(
                        f"El cluster no respondió en {self._command_timeout:.0f} "
                        "segundos; corté la espera."
                    ),
                    reason=CommandFailureReason.TIMEOUT,
                )
            exit_code = channel.recv_exit_status()
            return CommandResult(
                ok=exit_code == 0,
                stdout=stdout.read().decode(errors="replace"),
                stderr=stderr.read().decode(errors="replace"),
                # El comando corrió y dijo que no. Eso NO se reintenta: un
                # `permission denied` es igual de denegado la segunda vez.
                reason=None if exit_code == 0 else CommandFailureReason.COMMAND,
            )
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SSH: %s", exc)
            return CommandResult(
                ok=False,
                stderr=f"Error de conexión SSH: {exc}",
                reason=CommandFailureReason.TRANSPORT,
            )

    # ------------------------------------------------------------------
    # ClusterGateway
    # ------------------------------------------------------------------
    def submit_job(self, req: SlurmJobRequest) -> CommandResult:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={req.job_name}",
        ]
        # "default" es el relleno de la capa de aplicación cuando el usuario no
        # pidió partición: se omite la directiva y decide la default del cluster.
        if req.partition != "default":
            lines.append(f"#SBATCH --partition={req.partition}")
        # El cwd del job es el directorio del script: ahí quedan los
        # resultados Y el slurm-%j.out — que es lo que el monitor lee para
        # diagnosticar un fallo. script_path ya está validado (sin espacios
        # ni metacaracteres), puede ir directo en la directiva.
        run_dir = str(PurePosixPath(req.script_path).parent)
        lines += [
            f"#SBATCH --nodes={req.nodes}",
            f"#SBATCH --time={req.time_limit}",
            f"#SBATCH --chdir={run_dir}",
            f"bash {shlex.quote(req.script_path)}",
        ]
        script = "\n".join(lines)
        # sbatch acepta el script por stdin: evitamos archivo temporal remoto.
        command = f"sbatch <<'BECARIO_EOF'\n{script}\nBECARIO_EOF"
        # SIN reintento, y es lo más importante de este archivo: si el
        # `sbatch` llega al cluster y la respuesta se pierde, reintentar
        # encola el trabajo DOS veces. El usuario no ve el primero (no
        # tenemos su id) y paga las horas igual. Ante la duda, un envío que
        # falla y se avisa es infinitamente mejor que uno que se duplica en
        # silencio.
        result = self._run(command)
        if not result.ok:
            return result
        match = _SBATCH_JOB_ID_RE.search(result.stdout)
        job_id = match.group(1) if match else None
        if job_id is None:
            logger.warning("sbatch OK pero no pude parsear el job_id: %r", result.stdout)
        return CommandResult(ok=True, stdout=result.stdout, stderr=result.stderr, job_id=job_id)

    def cancel_job(self, job_id: JobId) -> CommandResult:
        # Reintentable: cancelar dos veces un trabajo ya cancelado no hace
        # nada. Es la diferencia con `sbatch`, que crea algo nuevo cada vez.
        return self._run(f"scancel {shlex.quote(job_id.value)}", reintentable=True)

    def job_status(self, job_id: Optional[JobId]) -> CommandResult:
        if job_id is not None:
            return self._run(
                "sacct -j "
                + shlex.quote(job_id.value)
                + " --format=JobID,JobName,State,Elapsed,ExitCode",
                reintentable=True,
            )
        return self._run(
            'squeue -u "$USER" --format="%.18i %.12j %.8T %.10M %.9P"',
            reintentable=True,
        )

    def job_state(self, job_id: JobId) -> JobStateReading:
        """Estado crudo de un trabajo, en una sola palabra — pensado para
        que el monitor lo interprete (ver `JobStatus.from_slurm`), no para
        mostrarlo directo al usuario.

        Devuelve `JobStateReading` y no `Optional[str]` porque el monitor
        necesita distinguir dos cosas que el `None` pelado juntaba: que
        `sacct` conteste y no conozca el trabajo (se lo purgó) y que no se
        haya podido preguntar (cluster caído). Con la segunda contando como
        «no hay noticias», un rato de red mala acercaba a los trabajos SANOS
        a que se los diera por perdidos.
        """
        result = self._run(
            "sacct -j "
            + shlex.quote(job_id.value)
            + " --format=State --noheader --parsable2 -X",
            reintentable=True,
        )
        if not result.ok:
            # Distinguir es todo el punto: un fallo del comando (sacct que
            # no existe, permisos) sí es una respuesta del cluster.
            return JobStateReading(state=None, reachable=not result.transitorio)
        if not result.stdout.strip():
            return JobStateReading(state=None, reachable=True)
        return JobStateReading(
            state=result.stdout.strip().splitlines()[0].strip() or None,
            reachable=True,
        )

    def job_exit_code(self, job_id: JobId) -> Optional[int]:
        """`ExitCode` viene como 'N:M' (salida:señal); interesa el primero.

        `-X` limita a la línea del trabajo (sin los pasos .batch/.extern),
        que es la que resume el desenlace.
        """
        result = self._run(
            "sacct -j "
            + shlex.quote(job_id.value)
            + " --format=ExitCode --noheader --parsable2 -X",
            reintentable=True,  # solo lee
        )
        if not result.ok or not result.stdout.strip():
            return None
        crudo = result.stdout.strip().splitlines()[0].strip()
        try:
            return int(crudo.split(":")[0])
        except (ValueError, IndexError):
            return None

    def make_directory(self, path: str) -> CommandResult:
        # `mkdir -p` es idempotente por definición: correrlo de nuevo sobre
        # un directorio que ya existe no hace nada ni falla.
        result = self._run(f"mkdir -p {shlex.quote(path)}", reintentable=True)
        if result.ok and not result.stdout.strip():
            # mkdir -p es silencioso al tener éxito: darle algo al usuario.
            # "listo" y no "creado": si ya existía, mkdir -p no crea nada.
            return CommandResult(ok=True, stdout=f"Directorio listo: {path}")
        return result

    def list_directory(self, path: str) -> CommandResult:
        # Vista de árbol de dos niveles. `tree` no está garantizado en
        # todos los clusters: si falta, se emula con find + indentación.
        result = self._run(
            f"tree -L 2 --noreport -- {shlex.quote(path)}", reintentable=True
        )
        if result.ok or "not found" not in result.stderr:
            return result
        # Sin pipe a propósito. `find <ruta> | sort` devuelve el código de
        # salida de `sort`, o sea 0 aunque la ruta no exista: el error se iba
        # por stderr y el listado vacío se dibujaba como un directorio vacío,
        # que es la única respuesta de todo el bot que directamente MIENTE.
        # El orden lo pone `_format_tree`, que no pierde nada al hacerlo.
        found = self._run(
            f"find {shlex.quote(path)} -mindepth 1 -maxdepth 2", reintentable=True
        )
        if not found.ok:
            return self._explicar_find_fallido(path, found)
        return CommandResult(
            ok=True,
            stdout=_format_tree(path, found.stdout),
            stderr=found.stderr,
        )

    def _explicar_find_fallido(self, path: str, fallo: CommandResult) -> CommandResult:
        """`find` falló: ¿la ruta no existe, o es otra cosa?

        Se pregunta por el directorio padre en vez de leer el stderr: el
        texto de `find` depende del locale del cluster, la lista de hermanos
        no. Y si no existe, los hermanos son la respuesta útil — las corridas
        llevan timestamp (`Zr_relajacion_20260819_225757`), así que el nombre
        que el usuario escribe de memoria casi nunca es el que está.
        """
        if fallo.transitorio:
            return fallo  # no se pudo mirar; no hay nada que concluir
        partes = PurePosixPath(path)
        padre, nombre = str(partes.parent), partes.name
        if not nombre or padre == path:
            return fallo
        vecinos = self._run(
            f"find {shlex.quote(padre)} -mindepth 1 -maxdepth 1", reintentable=True
        )
        if not vecinos.ok:
            return fallo
        hermanos = sorted({
            PurePosixPath(linea.strip()).name
            for linea in vecinos.stdout.splitlines() if linea.strip()
        })
        if nombre in hermanos:
            return fallo  # existe: `find` falló por otra cosa (permisos)
        return CommandResult(
            ok=False,
            stderr=f"No existe {path}. {_sugerir_hermanos(nombre, hermanos)}".strip(),
            reason=fallo.reason,
        )

    def upload_file(self, local_path: str, remote_path: str) -> CommandResult:
        """Sube un archivo por SFTP (reutiliza la conexión paramiko)."""
        try:
            client = self._connection()
            sftp = client.open_sftp()
            try:
                # Crear el directorio destino si no existe (mkdir -p simple)
                remote_dir = str(PurePosixPath(remote_path).parent)
                self._run(f"mkdir -p {shlex.quote(remote_dir)}", reintentable=True)
                sftp.put(local_path, remote_path)
            finally:
                sftp.close()
            return CommandResult(ok=True, stdout=f"Archivo subido a {remote_path}")
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP: %s", exc)
            return CommandResult(
                ok=False, stderr=f"Error subiendo archivo: {exc}",
                reason=CommandFailureReason.TRANSPORT,
            )

    def upload_dir(self, local_dir: str, remote_dir: str) -> CommandResult:
        """Sube un directorio completo por SFTP (recursivo)."""
        base = Path(local_dir)
        try:
            client = self._connection()
            sftp = client.open_sftp()
            try:
                for path in sorted(base.rglob("*")):
                    rel = path.relative_to(base).as_posix()
                    remote_path = f"{remote_dir.rstrip('/')}/{rel}"
                    if path.is_dir():
                        self._run(f"mkdir -p {shlex.quote(remote_path)}", reintentable=True)
                    else:
                        self._run(
                            f"mkdir -p {shlex.quote(str(PurePosixPath(remote_path).parent))}",
                            reintentable=True,
                        )
                        sftp.put(str(path), remote_path)
            finally:
                sftp.close()
            return CommandResult(ok=True, stdout=f"Directorio subido a {remote_dir}")
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP subiendo directorio: %s", exc)
            return CommandResult(
                ok=False, stderr=f"Error subiendo directorio: {exc}",
                reason=CommandFailureReason.TRANSPORT,
            )

    def home_dir(self) -> Optional[str]:
        """Home remoto de la cuenta (cacheado): sirve para volver absolutas
        las rutas de corrida cuando la base configurada es relativa."""
        if self._home_dir is None:
            result = self._run('printf "%s" "$HOME"', reintentable=True)
            if result.ok and result.stdout.strip().startswith("/"):
                self._home_dir = result.stdout.strip()
        return self._home_dir

    def file_exists(self, remote_path: str) -> Optional[bool]:
        """`True`/`False` si se pudo preguntar; `None` si no (ver el puerto).

        `FileNotFoundError` es una respuesta del servidor: el archivo no
        está. Un `SSHException`/`OSError` es otra cosa —no hubo conversación—
        y devolver `False` ahí equivalía a afirmar "no existe" sin haber
        podido mirar."""
        try:
            client = self._connection()
            sftp = client.open_sftp()
            try:
                sftp.stat(remote_path)
                return True
            finally:
                sftp.close()
        except FileNotFoundError:
            return False
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP (stat %s): %s", remote_path, exc)
            return None

    def list_dir(self, remote_dir: str) -> Optional[list[str]]:
        try:
            client = self._connection()
            sftp = client.open_sftp()
            try:
                return sorted(sftp.listdir(remote_dir))
            finally:
                sftp.close()
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP (listdir %s): %s", remote_dir, exc)
            return None

    def read_file(
        self, remote_path: str, max_bytes: Optional[int] = None
    ) -> Optional[str]:
        try:
            client = self._connection()
            sftp = client.open_sftp()
            try:
                with sftp.open(remote_path, "r") as handle:
                    # handle.read(None) lee todo (callers históricos con
                    # archivos chicos); con un tope solo baja ese prefijo,
                    # sin cargar en memoria un archivo potencialmente enorme.
                    return handle.read(max_bytes).decode(errors="replace")
            finally:
                sftp.close()
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP (read %s): %s", remote_path, exc)
            return None

    # ------------------------------------------------------------------
    # Corridas sin confirmar: mover al lugar bueno, o borrarlas
    # ------------------------------------------------------------------
    def move_run(self, src: str, dest: str) -> CommandResult:
        """Mueve una corrida de `.pending/` a su lugar definitivo.

        Un `mv` dentro del mismo sistema de archivos es atómico, así que la
        corrida aparece entera o no aparece: nunca a medias. Por eso se sube
        a un lado y se mueve, en vez de subir directo al destino — una
        subida cortada a la mitad deja un directorio que existe, parece
        válido y le falta el POTCAR.

        NO es reintentable: si el `mv` salió bien y la respuesta se perdió,
        el segundo intento falla con «no existe» e informaría un fallo
        falso sobre algo que sí pasó.
        """
        return self._run(f"mv -T {shlex.quote(src)} {shlex.quote(dest)}")

    def discard_pending(self, path: str) -> CommandResult:
        """Borra una corrida que quedó sin confirmar.

        El `rm -rf` es la única operación destructiva del gateway, así que
        va con su propio guard además del `shlex.quote`: solo se acepta una
        ruta absoluta, sin `..`, que esté DENTRO de un `.pending/` y no sea
        el `.pending/` mismo. La ruta la arma la aplicación, no el usuario
        ni el LLM — pero eso ya era cierto de todo lo demás, y el patrón de
        este proyecto es validar igual (ver el README, defensa en
        profundidad).
        """
        if not _es_ruta_de_pendientes(path):
            logger.error("Me negué a borrar una ruta fuera de pendientes: %r", path)
            return CommandResult(
                ok=False,
                stderr="Ruta fuera del área de pendientes; no borro nada.",
                reason=CommandFailureReason.COMMAND,
            )
        return self._run(f"rm -rf -- {shlex.quote(path)}", reintentable=True)

    def sweep_pending(self, pending_base: str, older_than_minutes: int) -> CommandResult:
        """Barre las corridas sin confirmar más viejas que N minutos.

        Es la red que cubre lo que el borrado explícito no puede: una
        confirmación que venció sin que nadie la tocara, y un bot que se
        reinició entre la subida y el botón. Barrer por EDAD no necesita
        llevar registro de nada — el estado está en el propio cluster.

        `-mindepth 1 -maxdepth 1` toca solo los hijos directos: nunca el
        contenedor. Si `.pending` no existe todavía, `find` sale con error
        y no pasa nada; por eso el `-o -true` del final no hace falta y el
        resultado se ignora río arriba.
        """
        if not _es_base_de_pendientes(pending_base):
            logger.error("Me negué a barrer una ruta que no es de pendientes: %r", pending_base)
            return CommandResult(
                ok=False,
                stderr="Ruta fuera del área de pendientes; no barro nada.",
                reason=CommandFailureReason.COMMAND,
            )
        quoted = shlex.quote(pending_base)
        return self._run(
            f"[ -d {quoted} ] && find {quoted} -mindepth 1 -maxdepth 1 "
            f"-mmin +{int(older_than_minutes)} -exec rm -rf -- {{}} + || true",
            reintentable=True,
        )

    def concat_files(self, sources: list[str], dest: str) -> CommandResult:
        """`cat` remoto: arma el POTCAR concatenando los de la biblioteca.
        Las rutas ya vienen validadas por el dominio; acá solo se quotean."""
        if not sources:
            return CommandResult(ok=False, stderr="No hay archivos para concatenar.")
        quoted = " ".join(shlex.quote(s) for s in sources)
        return self._run(f"cat {quoted} > {shlex.quote(dest)}")


class SSHClusterGatewayFactory:
    """Resuelve y cachea una conexión SSH por CUENTA del cluster, nunca por
    chat de Telegram: dos personas nunca comparten transporte ni sesión,
    cada una entra con su propia cuenta — así el aislamiento entre
    investigadores lo garantiza el propio sistema del cluster, no la app.
    """

    def __init__(
        self,
        default_host: str,
        default_port: int = 22,
        connect_timeout: float = 15.0,
        command_timeout: float = 120.0,
        keepalive_interval: float = 30.0,
    ) -> None:
        self._default_host = default_host
        self._default_port = default_port
        self._connect_timeout = connect_timeout
        self._command_timeout = command_timeout
        self._keepalive_interval = keepalive_interval
        self._cache: dict[str, SSHClusterGateway] = {}

    def for_identity(self, identity: ClusterIdentity) -> SSHClusterGateway:
        gateway = self._cache.get(identity.ssh_user)
        if gateway is None:
            gateway = SSHClusterGateway(
                host=identity.ssh_host or self._default_host,
                user=identity.ssh_user,
                key_path=identity.ssh_key_path,
                port=self._default_port,
                connect_timeout=self._connect_timeout,
                command_timeout=self._command_timeout,
                keepalive_interval=self._keepalive_interval,
            )
            self._cache[identity.ssh_user] = gateway
        return gateway

    def close_all(self) -> None:
        for gateway in self._cache.values():
            gateway.close()
        self._cache.clear()
