"""Monitoreo de trabajos enviados por BECARIO (cierra el loop).

Este servicio NO sabe nada de Telegram: produce `Notification`s neutras
(chat_id + texto) y deja que la presentación decida cómo enviarlas. La
programación periódica (cada cuánto correr `poll_and_notify`) vive en
`presentation/telegram_bot.py`, usando el `job_queue` de python-telegram-bot.
"""
from __future__ import annotations

import logging
import posixpath
import re
from dataclasses import dataclass
from typing import Optional

from ..domain.models import JobId, JobStatus
from ..domain.ports import (
    ClusterGateway,
    ClusterGatewayFactory,
    HistoryRepository,
    JobTracker,
    UserRegistry,
)

logger = logging.getLogger(__name__)

# Última energía E0 de un OSZICAR (una por paso iónico; para NSW=0 hay una sola).
_E0_RE = re.compile(r"E0=\s*([-+0-9.Ee]+)")
_ENCUT_DIR_RE = re.compile(r"\Aencut_(\d+)\Z")

# Criterio de convergencia estándar para el barrido de ENCUT.
_CONVERGENCE_MEV_PER_ATOM = 1.0

# Consultas SEGUIDAS en las que `sacct` contestó y NO conoció el trabajo,
# antes de darlo por perdido. Con el intervalo por defecto son cinco
# minutos.
#
# Estuvo en 60 (una hora) mientras `job_state()` devolvía el mismo `None`
# para «sacct no lo conoce» y para «el cluster no contesta»: con esa
# ambigüedad había que errar del lado de esperar, porque soltar un trabajo
# sano por un rato de red mala es peor que perseguir uno muerto. Ahora que
# `JobStateReading` separa los dos casos, esto cuenta solo respuestas
# EXPLÍCITAS del cluster diciendo que no lo conoce, y cinco de esas ya son
# concluyentes.
_MAX_CONSULTAS_SIN_RESPUESTA = 5


@dataclass(frozen=True)
class JobAck:
    """Lo que hay que asentar una vez que el aviso SALIÓ de verdad.

    Viaja con la notificación en vez de aplicarse al armarla: mientras el
    monitor marcaba el trabajo como notificado antes de que la presentación
    mandara el mensaje, un `send_message` fallido (Telegram caído, el
    usuario bloqueó al bot) se llevaba el aviso para siempre — el trabajo ya
    no volvía a aparecer en `active_jobs()` y nadie se enteraba de que había
    terminado. Justo el aviso que cierra el loop de todo el sistema.
    """

    job_id: str
    owner_id: int
    job_name: str
    estado: str


@dataclass(frozen=True)
class Notification:
    chat_id: int
    text: str
    # Pide fuente de ancho fijo (tablas), igual que `Reply.monospace`.
    monospace: bool = False
    # Solo la notificación PRINCIPAL de un trabajo lo trae. La cosecha del
    # barrido de ENCUT es un segundo mensaje del MISMO trabajo: volver a
    # asentarlo duplicaría la fila del historial.
    acuse: Optional[JobAck] = None


class JobMonitorService:
    """Un caso de uso más: revisar los trabajos activos y avisar a quien
    corresponda cuando terminan. Se apoya en los mismos puertos que
    `BecarioService` — misma identidad, misma conexión por cuenta."""

    def __init__(
        self,
        registry: UserRegistry,
        cluster_factory: ClusterGatewayFactory,
        tracker: JobTracker,
        history: HistoryRepository,
    ) -> None:
        self._registry = registry
        self._cluster_factory = cluster_factory
        self._tracker = tracker
        self._history = history

    def poll_and_notify(self) -> list[Notification]:
        notifications: list[Notification] = []
        for job in self._tracker.active_jobs():
            identity = self._registry.get_identity(job.owner_id)
            if identity is None:
                # Se dio de baja del roster entre el envío y ahora: dejamos
                # de rastrearlo, no hay a quién avisarle.
                logger.info(
                    "Job %s de un usuario ya no registrado (%s); se deja de rastrear.",
                    job.job_id, job.owner_id,
                )
                self._tracker.mark_notified(job.job_id, job.owner_id)
                continue

            cluster = self._cluster_factory.for_identity(identity)
            lectura = cluster.job_state(JobId(value=job.job_id))
            if not lectura.reachable:
                # No hubo conversación con el cluster. NO cuenta para la
                # racha: el trabajo no tiene la culpa de que se haya caído
                # la red, y contarlo acercaba a los trabajos SANOS a que se
                # los diera por perdidos.
                logger.warning(
                    "No pude hablar con el cluster para consultar %s; reintento la próxima vuelta.",
                    job.job_id,
                )
                continue
            raw_state = lectura.state
            if raw_state is None:
                # `sacct` contestó y no lo conoce. ESO sí cuenta.
                aviso = self._sin_noticias(job)
                if aviso is not None:
                    notifications.append(aviso)
                continue
            if job.poll_attempts:
                # Volvió a contestar: la racha se corta. Se cuentan
                # consultas SEGUIDAS, no acumuladas — un SSH que se cayó
                # una vez no puede acercar al trabajo a darse por perdido.
                self._tracker.clear_unreachable(job.job_id, job.owner_id)

            new_status = JobStatus.from_slurm(raw_state)
            if new_status != job.status:
                self._tracker.update_status(job.job_id, job.owner_id, new_status)

            if new_status not in JobStatus.terminal():
                continue  # sigue en cola o corriendo: se revisa de nuevo la próxima vuelta

            icon = "✅" if new_status is JobStatus.COMPLETED else "⚠️"
            text = (
                f"{icon} Tu trabajo {job.job_id} ({job.job_name}) terminó: "
                f"{new_status.label_es}."
            )
            if job.script_path:
                # El directorio del script es donde quedaron los resultados
                # (los scripts de cálculo hacen cd a su propio directorio).
                run_dir = posixpath.dirname(job.script_path)
                text += f"\n📂 Corrida en: {run_dir}"
                if new_status in (JobStatus.FAILED, JobStatus.TIMEOUT):
                    text += self._failure_diagnostics(cluster, run_dir, job.job_id)
            # Ni `history.add` ni `mark_notified` acá: los dos van en el
            # acuse, cuando el mensaje haya salido. Asentarlos ahora es
            # afirmar que el usuario se enteró antes de habérselo dicho.
            notifications.append(Notification(
                chat_id=job.chat_id,
                text=text,
                acuse=JobAck(
                    job_id=job.job_id, owner_id=job.owner_id,
                    job_name=job.job_name, estado=new_status.label_es,
                ),
            ))

            # Workflows: al terminar bien un barrido de ENCUT, cosechar las
            # energías del cluster y mandar la curva como segundo mensaje.
            if (
                job.workflow == "encut_scan"
                and new_status is JobStatus.COMPLETED
                and job.script_path
            ):
                harvest = self._harvest_encut_scan(
                    cluster, posixpath.dirname(job.script_path), job.job_name
                )
                if harvest is not None:
                    notifications.append(
                        Notification(
                            chat_id=job.chat_id,
                            text=harvest.text,
                            monospace=harvest.monospace,
                        )
                    )

        return notifications

    # ------------------------------------------------------------------
    # Acuse de entrega
    # ------------------------------------------------------------------
    def confirm_delivery(self, ack: JobAck) -> None:
        """Asienta el trabajo DESPUÉS de que su aviso salió.

        Lo llama la presentación, que es la única que sabe si el mensaje
        llegó a Telegram. Hasta que esto corra, el trabajo sigue en
        `active_jobs()` y el próximo tick lo reintenta.

        La entrega queda así en «al menos una vez»: si el mensaje sale pero
        este asiento falla (base bloqueada), el próximo tick vuelve a
        avisar. Es el lado correcto para equivocarse — que a alguien le
        lleguen dos veces «tu trabajo terminó» es una molestia; que no le
        llegue ninguna es el bug que este método existe para cerrar.
        """
        self._history.add(
            owner_id=ack.owner_id, job_id=ack.job_id,
            nombre_trabajo=ack.job_name, estado=ack.estado,
        )
        self._tracker.mark_notified(ack.job_id, ack.owner_id)

    # ------------------------------------------------------------------
    # Trabajos de los que ya no hay noticias
    # ------------------------------------------------------------------
    def _sin_noticias(self, job) -> Optional[Notification]:
        """Qué hacer cuando no se puede leer el estado de un trabajo.

        Hasta acá esto era un `warning` y un «reintento la próxima vuelta»
        sin final: un trabajo que Slurm ya purgó de `sacct` (`MinJobAge`)
        se consultaba cada 60 segundos para siempre, gastando una ida y
        vuelta SSH por vuelta y sin que nadie se enterara.

        Se cuenta la racha y no la antigüedad a propósito. La antigüedad
        parece el criterio obvio y es el equivocado: un trabajo puede estar
        legítimamente encolado durante días, así que «viejo» no distingue
        «perdido» de «esperando turno». Que no conteste, sí.
        """
        intentos = self._tracker.record_unreachable(job.job_id, job.owner_id)
        if intentos < _MAX_CONSULTAS_SIN_RESPUESTA:
            logger.warning(
                "No pude consultar el estado de %s (%s seguidas); reintento la próxima vuelta.",
                job.job_id, intentos,
            )
            return None
        logger.error(
            "Dejo de rastrear %s: %s consultas seguidas sin poder leer su estado.",
            job.job_id, intentos,
        )
        return Notification(
            chat_id=job.chat_id,
            text=(
                f"🔎 Perdí el rastro del trabajo {job.job_id} ({job.job_name}): "
                f"llevo {intentos} consultas seguidas sin poder leer su estado, "
                "así que dejo de seguirlo.\n\nLo más probable es que Slurm ya lo "
                "haya sacado de `sacct`. Si creés que sigue corriendo, "
                "consultalo con «estado de mis trabajos»."
            ),
            acuse=JobAck(
                job_id=job.job_id, owner_id=job.owner_id,
                job_name=job.job_name, estado="sin rastro",
            ),
        )

    # ------------------------------------------------------------------
    # Diagnóstico de fallos: leer los logs remotos de la corrida
    # ------------------------------------------------------------------
    def _failure_diagnostics(
        self, cluster: ClusterGateway, run_dir: str, job_id: str
    ) -> str:
        """Colas de los logs de la corrida para que el aviso de fallo diga
        POR QUÉ falló, no solo que falló."""
        sections: list[str] = []

        slurm_out = cluster.read_file(f"{run_dir}/slurm-{job_id}.out")
        if slurm_out and slurm_out.strip():
            sections.append(f"— slurm-{job_id}.out:\n{_tail(slurm_out)}")

        # vasp.out de la raíz (cálculo simple) o del último punto que llegó
        # a correr (barrido).
        vasp_out = cluster.read_file(f"{run_dir}/vasp.out")
        vasp_label = "vasp.out"
        if vasp_out is None:
            entries = cluster.list_dir(run_dir) or []
            for name in sorted(
                (e for e in entries if _ENCUT_DIR_RE.match(e)),
                key=lambda e: int(e.split("_")[1]),
                reverse=True,
            ):
                vasp_out = cluster.read_file(f"{run_dir}/{name}/vasp.out")
                if vasp_out is not None:
                    vasp_label = f"{name}/vasp.out"
                    break
        if vasp_out and vasp_out.strip():
            sections.append(f"— {vasp_label}:\n{_tail(vasp_out)}")

        if not sections:
            sections.append(self._sin_salida(cluster, job_id))
        return "\n🔍 Diagnóstico:\n" + "\n".join(sections)

    # Códigos de salida que dicen algo concreto. Un 127 manda a mirar el
    # PATH del nodo; un 1 de VASP manda a mirar el INCAR. Son búsquedas
    # distintas, y antes las dos recibían la misma conjetura.
    _CODIGOS = {
        126: "el script existe pero no se pudo ejecutar (¿permisos?, ¿falta el shebang?)",
        127: "comando o script no encontrado en el nodo (revisá el PATH, "
             "los `module load` del prelude, o si el directorio de la corrida "
             "se ve desde los nodos de cómputo)",
        137: "lo mató una señal 9: casi siempre falta de memoria",
        139: "violación de segmento en el binario",
    }

    def _sin_salida(self, cluster: ClusterGateway, job_id: str) -> str:
        """Qué decir cuando la corrida no dejó ningún archivo de salida.

        Antes esto afirmaba, con seguridad, que el script nunca había
        llegado a ejecutarse. A veces acertaba —y en el caso que motivó
        el mensaje, acertó— pero lo hacía por INFERENCIA desde la ausencia
        de archivos, con el código de salida a un `sacct` de distancia y
        sin consultarlo. Una hipótesis enunciada con esa seguridad manda a
        investigar al lugar equivocado cuando falla.
        """
        code = cluster.job_exit_code(JobId(value=job_id))
        if code is None:
            return (
                "No encontré archivos de salida ni pude leer el código de "
                "salida del trabajo, así que no puedo decirte por qué falló."
            )
        explicacion = self._CODIGOS.get(code)
        detalle = f": {explicacion}" if explicacion else " (sin una causa que sepa traducir)"
        return (
            f"No encontré archivos de salida en la corrida. Slurm reporta "
            f"código de salida {code}{detalle}."
        )

    # ------------------------------------------------------------------
    # Cosecha del barrido de ENCUT (sin estado local: relee el cluster)
    # ------------------------------------------------------------------
    def _harvest_encut_scan(
        self, cluster: ClusterGateway, run_dir: str, job_name: str
    ) -> Optional[Notification]:
        """Lee los `encut_*/OSZICAR` de la corrida y arma la tabla E(ENCUT)
        con el ENCUT recomendado. Devuelve una notificación sin chat_id útil
        (solo texto+monospace); el llamador le pone el chat."""
        entries = cluster.list_dir(run_dir)
        if entries is None:
            return Notification(
                chat_id=0,
                text=f"⚠️ No pude leer el directorio de la corrida ({run_dir}) "
                "para armar la curva de ENCUT.",
            )
        scan_dirs = sorted(
            (e for e in entries if _ENCUT_DIR_RE.match(e)),
            key=lambda name: int(name.split("_")[1]),
        )
        if not scan_dirs:
            return Notification(
                chat_id=0,
                text=f"⚠️ No encontré subdirectorios encut_* en {run_dir}.",
            )

        n_atoms = _parse_poscar_n_atoms(
            cluster.read_file(f"{run_dir}/{scan_dirs[0]}/POSCAR")
        )

        points: list[tuple[int, float]] = []
        missing: list[str] = []
        for name in scan_dirs:
            energy = _parse_last_e0(cluster.read_file(f"{run_dir}/{name}/OSZICAR"))
            if energy is None:
                missing.append(name)
            else:
                points.append((int(name.split("_")[1]), energy))

        if len(points) < 2:
            return Notification(
                chat_id=0,
                text=(
                    f"⚠️ El barrido terminó pero no pude leer energías suficientes "
                    f"en {run_dir} (revisá los vasp.out de {', '.join(missing) or 'los puntos'})."
                ),
            )

        e_ref = points[-1][1]  # energía al ENCUT más alto
        per_atom = n_atoms or 1
        rows = [
            (encut, energy, abs(energy - e_ref) * 1000.0 / per_atom)
            for encut, energy in points
        ]
        recommended = next(
            (
                encut
                for encut, _, delta in rows[:-1]
                if delta < _CONVERGENCE_MEV_PER_ATOM
            ),
            None,
        )

        unit = "meV/át" if n_atoms else "meV"
        header = f"{'ENCUT':>6}  {'E (eV)':>14}  {'ΔE (' + unit + ')':>12}"
        lines = [f"📈 Convergencia de ENCUT ({job_name}):", "", header,
                 "-" * len(header)]
        lines += [
            f"{encut:>6}  {energy:>14.6f}  {delta:>12.2f}"
            for encut, energy, delta in rows
        ]
        if missing:
            lines.append(f"(sin datos: {', '.join(missing)})")
        lines.append("")
        if recommended is not None:
            lines.append(
                f"✅ ENCUT recomendado: {recommended} eV "
                f"(ΔE < {_CONVERGENCE_MEV_PER_ATOM:g} {unit} respecto de {rows[-1][0]} eV)"
            )
        else:
            lines.append(
                f"⚠️ Ningún punto converge a menos de "
                f"{_CONVERGENCE_MEV_PER_ATOM:g} {unit} del máximo: "
                f"conviene extender el barrido por encima de {rows[-1][0]} eV."
            )
        return Notification(chat_id=0, text="\n".join(lines), monospace=True)


def _tail(text: str, n_lines: int = 10, max_chars: int = 700) -> str:
    """Últimas líneas de un log, acotadas para caber en un mensaje."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    clipped = "\n".join(lines[-n_lines:])
    if len(clipped) > max_chars:
        clipped = "…" + clipped[-max_chars:]
    return clipped


def _parse_last_e0(oszicar: Optional[str]) -> Optional[float]:
    if not oszicar:
        return None
    matches = _E0_RE.findall(oszicar)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _parse_poscar_n_atoms(poscar: Optional[str]) -> Optional[int]:
    """Total de átomos de un POSCAR: la línea de cantidades es la 7ª (VASP 5,
    con símbolos en la 6ª) o la 6ª (VASP 4)."""
    if not poscar:
        return None
    lines = poscar.splitlines()
    for idx in (6, 5):
        if len(lines) > idx:
            try:
                return sum(int(x) for x in lines[idx].split())
            except ValueError:
                continue
    return None
