"""Adaptador SSH hacia el cluster (implementa ClusterGateway).

Regla de oro: cada argumento que viaja al shell remoto pasa por
`shlex.quote`. Los métodos reciben modelos de dominio ya validados,
así que hay defensa en profundidad: validación semántica (Pydantic)
+ quoting sintáctico (shlex).
"""
from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Optional

import paramiko

from ..domain.models import ClusterIdentity, CommandResult, JobId, SlurmJobRequest

logger = logging.getLogger(__name__)

_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


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
    ) -> None:
        self._host = host
        self._user = user
        self._key_path = key_path
        self._port = port
        self._connect_timeout = connect_timeout
        self._command_timeout = command_timeout
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
        self._client = client
        return client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _run(self, command: str) -> CommandResult:
        try:
            client = self._connection()
            _, stdout, stderr = client.exec_command(
                command, timeout=self._command_timeout
            )
            exit_code = stdout.channel.recv_exit_status()
            return CommandResult(
                ok=exit_code == 0,
                stdout=stdout.read().decode(errors="replace"),
                stderr=stderr.read().decode(errors="replace"),
            )
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SSH: %s", exc)
            return CommandResult(ok=False, stderr=f"Error de conexión SSH: {exc}")

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
        lines += [
            f"#SBATCH --nodes={req.nodes}",
            f"#SBATCH --time={req.time_limit}",
            f"bash {shlex.quote(req.script_path)}",
        ]
        script = "\n".join(lines)
        # sbatch acepta el script por stdin: evitamos archivo temporal remoto.
        command = f"sbatch <<'BECARIO_EOF'\n{script}\nBECARIO_EOF"
        result = self._run(command)
        if not result.ok:
            return result
        match = _SBATCH_JOB_ID_RE.search(result.stdout)
        job_id = match.group(1) if match else None
        if job_id is None:
            logger.warning("sbatch OK pero no pude parsear el job_id: %r", result.stdout)
        return CommandResult(ok=True, stdout=result.stdout, stderr=result.stderr, job_id=job_id)

    def cancel_job(self, job_id: JobId) -> CommandResult:
        return self._run(f"scancel {shlex.quote(job_id.value)}")

    def job_status(self, job_id: Optional[JobId]) -> CommandResult:
        if job_id is not None:
            return self._run(
                "sacct -j "
                + shlex.quote(job_id.value)
                + " --format=JobID,JobName,State,Elapsed,ExitCode"
            )
        return self._run('squeue -u "$USER" --format="%.18i %.12j %.8T %.10M %.9P"')

    def job_state(self, job_id: JobId) -> Optional[str]:
        """Estado crudo de un trabajo, en una sola palabra — pensado para
        que el monitor lo interprete (ver `JobStatus.from_slurm`), no para
        mostrarlo directo al usuario."""
        result = self._run(
            "sacct -j "
            + shlex.quote(job_id.value)
            + " --format=State --noheader --parsable2 -X"
        )
        if not result.ok or not result.stdout.strip():
            return None
        return result.stdout.strip().splitlines()[0].strip() or None

    def upload_file(self, local_path: str, remote_path: str) -> CommandResult:
        """Sube un archivo por SFTP (reutiliza la conexión paramiko)."""
        try:
            client = self._connection()
            sftp = client.open_sftp()
            try:
                # Crear el directorio destino si no existe (mkdir -p simple)
                remote_dir = str(PurePosixPath(remote_path).parent)
                self._run(f"mkdir -p {shlex.quote(remote_dir)}")
                sftp.put(local_path, remote_path)
            finally:
                sftp.close()
            return CommandResult(ok=True, stdout=f"Archivo subido a {remote_path}")
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP: %s", exc)
            return CommandResult(ok=False, stderr=f"Error subiendo archivo: {exc}")

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
                        self._run(f"mkdir -p {shlex.quote(remote_path)}")
                    else:
                        self._run(f"mkdir -p {shlex.quote(str(PurePosixPath(remote_path).parent))}")
                        sftp.put(str(path), remote_path)
            finally:
                sftp.close()
            return CommandResult(ok=True, stdout=f"Directorio subido a {remote_dir}")
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP subiendo directorio: %s", exc)
            return CommandResult(ok=False, stderr=f"Error subiendo directorio: {exc}")

    def home_dir(self) -> Optional[str]:
        """Home remoto de la cuenta (cacheado): sirve para volver absolutas
        las rutas de corrida cuando la base configurada es relativa."""
        if self._home_dir is None:
            result = self._run('printf "%s" "$HOME"')
            if result.ok and result.stdout.strip().startswith("/"):
                self._home_dir = result.stdout.strip()
        return self._home_dir

    def file_exists(self, remote_path: str) -> bool:
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
            return False

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

    def read_file(self, remote_path: str) -> Optional[str]:
        try:
            client = self._connection()
            sftp = client.open_sftp()
            try:
                with sftp.open(remote_path, "r") as handle:
                    return handle.read().decode(errors="replace")
            finally:
                sftp.close()
        except (paramiko.SSHException, OSError) as exc:
            logger.error("Fallo SFTP (read %s): %s", remote_path, exc)
            return None

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
    ) -> None:
        self._default_host = default_host
        self._default_port = default_port
        self._connect_timeout = connect_timeout
        self._command_timeout = command_timeout
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
            )
            self._cache[identity.ssh_user] = gateway
        return gateway

    def close_all(self) -> None:
        for gateway in self._cache.values():
            gateway.close()
        self._cache.clear()
