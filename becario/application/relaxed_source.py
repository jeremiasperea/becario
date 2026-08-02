"""Arrancar un cálculo desde la estructura YA RELAJADA de una corrida previa.

Pedir "el bulk de ZrO2 relajado" es pedir el CONTCAR de la última relajación
de ese material. Todo el módulo existe por una razón: un CONTCAR puede estar
ausente, a medio escribir, o —el caso peligroso— completo, parseable y NO
convergido. Ese último caso no falla en ningún lado: la corrida termina con
código 0, Slurm dice COMPLETED, el archivo abre perfecto, y la estructura no
es la relajada. Partir de ahí envenena en silencio todo lo que venga después.

Por eso la resolución es FAIL-CLOSED y ocurre ANTES de crear ningún
directorio de corrida (mismo principio que `_resolve_structure` con Materials
Project): si algo no está en condiciones, no se arma nada y se avisa qué
falta. La única excepción es la convergencia, que avisa pero deja decidir —
a veces una estructura a medio relajar es un punto de partida perfectamente
razonable, y esa es una decisión de quien hace la física, no del bot.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..domain.models import CalcKind, JobId, JobStatus
from ..domain.vasp_tags import cita
from ..domain.ports import ClusterGateway

if TYPE_CHECKING:  # pragma: no cover - solo para anotar
    from ase import Atoms

logger = logging.getLogger(__name__)

# Un paso iónico por línea con "F=" en el OSZICAR. Es el conteo que se compara
# contra NSW para saber si la relajación convergió o se quedó sin presupuesto.
_IONIC_STEP_RE = re.compile(r"^\s*\d+\s+F=", re.MULTILINE)
_NSW_RE = re.compile(r"^\s*NSW\s*=\s*(\d+)", re.MULTILINE | re.IGNORECASE)

# Topes de lectura: el INCAR es diminuto y el OSZICAR crece con los pasos
# iónicos (una línea por paso electrónico). `read_file` baja solo el prefijo,
# así que esto acota cuánto viaja por SFTP.
_INCAR_MAX_BYTES = 8_000
_OSZICAR_MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class RelaxedStructure:
    """CONTCAR resuelto y listo para usar como punto de partida."""

    atoms: "Atoms"
    run_dir: str
    job_name: str
    # None => no se pudo determinar la convergencia (falta OSZICAR o INCAR).
    # True/False => se pudo, y `warning` cuenta el caso.
    converged: Optional[bool]
    warning: str = ""

    def note(self) -> str:
        head = f"♻️ Estructura de partida: CONTCAR de {self.job_name}\n📂 {self.run_dir}"
        return f"{head}\n{self.warning}" if self.warning else head


class RelaxedSourceError(Exception):
    """No se puede partir de una relajación previa. `message` ya está
    redactado para mostrarle al usuario: dice QUÉ falta, no un stacktrace."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def resolve_relaxed_structure(
    calc_runs, cluster: ClusterGateway, owner_id: int, formula: str
) -> RelaxedStructure:
    """Última relajación de `formula` de ESTE usuario, como `Atoms`.

    Lanza `RelaxedSourceError` —sin efectos, sin nada creado— si no hay de
    dónde partir. Los chequeos van en orden de lo más barato a lo más caro y
    de lo más informativo a lo menos: saber que el trabajo sigue corriendo es
    mejor mensaje que "no encontré el CONTCAR".
    """
    run = _latest_relaxation(calc_runs, owner_id, formula)
    run_dir = str(run["run_dir"]).rstrip("/")
    job_name = str(run["job_name"])

    _reject_if_unfinished(cluster, run, job_name)

    contcar_path = f"{run_dir}/CONTCAR"
    exists = cluster.file_exists(contcar_path)
    if exists is None:
        # No se pudo preguntar. Decir "no tiene CONTCAR" mandaría a mirar una
        # corrida que puede estar perfecta, cuando el problema es el acceso.
        raise RelaxedSourceError(
            f"⚠️ No pude consultar {run_dir}: no llegué al cluster. No sé si "
            f"la corrida {job_name} tiene CONTCAR. No armé nada."
        )
    if not exists:
        raise RelaxedSourceError(
            f"⚠️ La corrida {job_name} no tiene CONTCAR en {run_dir}. "
            "Puede que VASP nunca haya llegado a escribirlo. No armé nada."
        )

    atoms = _parse_contcar(cluster, contcar_path, job_name)
    converged, warning = _check_convergence(cluster, run_dir, job_name)
    return RelaxedStructure(
        atoms=atoms, run_dir=run_dir, job_name=job_name,
        converged=converged, warning=warning,
    )


def _latest_relaxation(calc_runs, owner_id: int, formula: str) -> dict:
    if calc_runs is None:
        raise RelaxedSourceError(
            "⚠️ No tengo registro de corridas previas, así que no puedo partir "
            "de una estructura relajada."
        )
    prefix = f"{formula}_{CalcKind.RELAX.value}"
    runs = calc_runs.find_recent(owner_id, prefix, limit=1)
    if not runs:
        raise RelaxedSourceError(
            f"⚠️ No encontré ninguna relajación previa de {formula} tuya. "
            "Primero hay que relajarlo; después puedo partir de ese resultado."
        )
    return runs[0]


def _reject_if_unfinished(cluster: ClusterGateway, run: dict, job_name: str) -> None:
    """Corriendo o terminada mal => no se toca nada."""
    job_id = str(run.get("job_id") or "").strip()
    if not job_id:
        return  # sin id no se puede consultar; los chequeos de archivo deciden
    status = JobStatus.from_slurm(cluster.job_state(JobId(value=job_id)))
    if status is JobStatus.UNKNOWN:
        # Slurm ya no la recuerda: normal en corridas viejas. Los chequeos de
        # archivo (CONTCAR + convergencia) alcanzan para decidir.
        return
    if status not in JobStatus.terminal():
        raise RelaxedSourceError(
            f"⏳ La relajación {job_name} ({job_id}) todavía está {status.label_es}. "
            "Esperá a que termine y volvé a pedírmelo — no armé nada."
        )
    if status is not JobStatus.COMPLETED:
        raise RelaxedSourceError(
            f"⚠️ La relajación {job_name} ({job_id}) {status.label_es}. "
            "Su CONTCAR no sirve como punto de partida. No armé nada."
        )


def _parse_contcar(
    cluster: ClusterGateway, path: str, job_name: str
) -> "Atoms":
    from ase.io import read  # import perezoso: el dominio no depende de ASE

    text = cluster.read_file(path)
    if not text or not text.strip():
        raise RelaxedSourceError(
            f"⚠️ El CONTCAR de {job_name} está vacío. No armé nada."
        )
    try:
        return read(io.StringIO(text), format="vasp")
    except Exception as exc:
        # Típico de un CONTCAR a medio escribir.
        logger.warning("CONTCAR ilegible en %s: %s", path, exc)
        raise RelaxedSourceError(
            f"⚠️ No pude leer el CONTCAR de {job_name}: está incompleto o "
            "corrupto. No armé nada."
        ) from exc


def _check_convergence(
    cluster: ClusterGateway, run_dir: str, job_name: str
) -> tuple[Optional[bool], str]:
    """¿La relajación alcanzó el criterio, o se quedó sin pasos iónicos?

    Se compara el conteo de pasos del OSZICAR contra el NSW del INCAR en vez
    de buscar "reached required accuracy" en el OUTCAR: esa marca está al
    FINAL del archivo y `read_file` solo baja un prefijo, así que habría que
    ampliar el puerto para leer la cola. Si los pasos llegaron al tope, la
    corrida se quedó sin presupuesto, que es exactamente el caso peligroso.
    """
    incar = cluster.read_file(f"{run_dir}/INCAR", max_bytes=_INCAR_MAX_BYTES)
    oszicar = cluster.read_file(f"{run_dir}/OSZICAR", max_bytes=_OSZICAR_MAX_BYTES)
    if not incar or not oszicar:
        return None, (
            "⚠️ No pude verificar si esa relajación convergió "
            f"(falta INCAR u OSZICAR en {run_dir}). Revisala antes de confiar "
            "en la estructura."
        )
    if len(oszicar) >= _OSZICAR_MAX_BYTES:
        # `read_file` baja solo un PREFIJO: si el OSZICAR llegó al tope, los
        # pasos que faltan son justo los del final. Contar sobre eso daría
        # MENOS pasos que NSW y diría "convergió" — el error hacia el lado
        # peligroso. Preferimos declararlo no verificable.
        return None, (
            f"⚠️ El OSZICAR de {job_name} es demasiado grande para verificar "
            "la convergencia desde acá. Revisá a mano si la relajación llegó "
            "al criterio antes de confiar en esta estructura."
        )
    nsw_match = _NSW_RE.search(incar)
    if not nsw_match:
        return None, (
            "⚠️ No pude verificar si esa relajación convergió: el INCAR no "
            f"declara NSW{cita('NSW')}."
        )

    nsw = int(nsw_match.group(1))
    steps = len(_IONIC_STEP_RE.findall(oszicar))
    if nsw > 0 and steps >= nsw:
        return False, (
            f"⚠️ OJO: {job_name} usó los {steps} pasos iónicos que tenía "
            f"asignados{cita('NSW')} sin alcanzar el criterio de fuerzas"
            f"{cita('EDIFFG')}. Terminó sin error, pero esa estructura NO "
            "está relajada del todo. Vos decidís si igual querés partir de ahí."
        )
    return True, ""
