"""Monitoreo de trabajos enviados por BECARIO (cierra el loop).

Este servicio NO sabe nada de Telegram: produce `Notification`s neutras
(chat_id + texto) y deja que la presentación decida cómo enviarlas. La
programación periódica (cada cuánto correr `poll_and_notify`) vive en
`presentation/telegram_bot.py`, usando el `job_queue` de python-telegram-bot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..domain.models import JobId, JobStatus
from ..domain.ports import ClusterGatewayFactory, HistoryRepository, JobTracker, UserRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Notification:
    chat_id: int
    text: str


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
                f"{new_status.value.lower()}"
            )
            self._history.add(
                owner_id=job.owner_id, job_id=job.job_id,
                nombre_trabajo=job.job_name, estado=new_status.value,
            )
            self._tracker.mark_notified(job.job_id, job.owner_id)
            notifications.append(Notification(chat_id=job.chat_id, text=text))

        return notifications
