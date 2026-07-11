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


@dataclass(frozen=True)
class Notification:
    chat_id: int
    text: str
    # Pide fuente de ancho fijo (tablas), igual que `Reply.monospace`.
    monospace: bool = False


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
            raw_state = cluster.job_state(JobId(value=job.job_id))
            if raw_state is None:
                logger.warning("No pude consultar el estado de %s; reintento la próxima vuelta.", job.job_id)
                continue

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
                text += f"\n📂 Corrida en: {posixpath.dirname(job.script_path)}"
            self._history.add(
                owner_id=job.owner_id, job_id=job.job_id,
                nombre_trabajo=job.job_name, estado=new_status.label_es,
            )
            self._tracker.mark_notified(job.job_id, job.owner_id)
            notifications.append(Notification(chat_id=job.chat_id, text=text))

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
