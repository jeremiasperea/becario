"""Registro de identidades del grupo (implementa UserRegistry).

No hay altas/bajas desde Telegram — no existe un rol admin dentro del bot.
Quien administra el grupo edita el roster con `scripts/manage_users.py`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from ..domain.models import ClusterIdentity

logger = logging.getLogger(__name__)


class JSONUserRegistry:
    """Roster cargado desde un archivo JSON:

    {"users": [{"telegram_user_id": 111, "ssh_user": "jperez",
                "ssh_key_path": "/home/becario/.ssh/id_jperez",
                "display_name": "Juan Pérez"}]}
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._by_user_id: dict[int, ClusterIdentity] = {}
        self.reload()

    def reload(self) -> None:
        if not self._path.exists():
            logger.warning(
                "Roster no encontrado en %s; nadie está registrado todavía.",
                self._path,
            )
            self._by_user_id = {}
            return

        raw = json.loads(self._path.read_text(encoding="utf-8"))
        by_id: dict[int, ClusterIdentity] = {}
        for entry in raw.get("users", []):
            try:
                identity = ClusterIdentity(**entry)
            except ValidationError as exc:
                logger.error("Entrada de roster inválida, se ignora: %s (%s)", entry, exc)
                continue
            by_id[identity.telegram_user_id] = identity
        self._by_user_id = by_id
        logger.info("Roster cargado: %d usuarios.", len(by_id))

    def get_identity(self, telegram_user_id: int) -> Optional[ClusterIdentity]:
        return self._by_user_id.get(telegram_user_id)
