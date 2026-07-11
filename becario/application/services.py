"""Casos de uso de B.E.C.A.R.I.O.

Esta capa orquesta el dominio y los puertos. No conoce Telegram, paramiko
ni Ollama: recibe texto, devuelve objetos `Reply`. La presentación decide
cómo renderizarlos (mensaje simple o botones de confirmación).

Modelo multiusuario: no existe un rol admin dentro del bot. Cada mensaje
se resuelve a la `ClusterIdentity` de quien lo mandó (vía `UserRegistry`)
y las operaciones sobre el cluster corren con SU propia cuenta SSH
(vía `ClusterGatewayFactory`) — el aislamiento entre personas lo garantiza
Slurm/el sistema operativo del cluster, no lógica de la aplicación.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from ..domain.models import (
    CalcKind,
    ClusterIdentity,
    HistoryFilter,
    Intent,
    JobId,
    OutputFormat,
    PendingAction,
    SlurmJobRequest,
    StructureKind,
    StructureRequest,
    TrackedJob,
    VaspCalcRequest,
)
from ..domain.ports import (
    CalcInputGenerator,
    ClusterGateway,
    ClusterGatewayFactory,
    ConfirmationStore,
    HistoryRepository,
    IntentRouter,
    JobTracker,
    StructureBuilder,
    UserRegistry,
)

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "❓ No pude interpretar tu pedido. Probá con:\n"
    '• "Relajá los parámetros de red del bulk de W"\n'
    '• "Hacé la curva de convergencia de ENCUT para Zr hcp"\n'
    '• "Mostrá el estado de mis trabajos"\n'
    '• "Cancelá el trabajo 12345"\n'
    '• "Consultá el historial"\n'
    '• "Generá un POSCAR de Si diamond 2x2x2"'
)

NOT_REGISTERED_TEXT = (
    "🚫 No estás registrado en B.E.C.A.R.I.O. Pedile a quien administra "
    "el grupo que te agregue con tu usuario del cluster."
)

FOREIGN_CONFIRMATION_TEXT = "⚠️ Esta confirmación no te pertenece."
EXPIRED_CONFIRMATION_TEXT = "⌛ Esta confirmación expiró o ya fue usada."


@dataclass(frozen=True)
class Reply:
    """Respuesta neutral respecto del canal de salida."""

    text: str
    needs_confirmation: bool = False
    confirmation_token: Optional[str] = None
    # Pide fuente de ancho fijo (tablas): cada canal decide cómo lograrlo.
    monospace: bool = False


@dataclass(frozen=True)
class _Ctx:
    """Contexto resuelto de un pedido: quién es y con qué cuenta opera."""

    chat_id: int
    user_id: int
    identity: ClusterIdentity
    cluster: ClusterGateway


def _format_history_table(rows: list[dict]) -> str:
    """Tabla de ancho fijo con el historial. Pensada para mostrarse con
    fuente monoespaciada (ver `Reply.monospace`)."""
    headers = ("Fecha", "Job", "Nombre", "Estado")
    table = [
        (
            str(r.get("fecha", ""))[:16],  # sin segundos, ocupa menos
            str(r.get("job_id", "")),
            str(r.get("nombre_trabajo", "")) or "-",
            str(r.get("estado", "")),
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(headers)]

    def fila(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    separador = "-" * (sum(widths) + 2 * (len(widths) - 1))
    return "\n".join([fila(headers), separador] + [fila(t) for t in table])


class BecarioService:
    """Fachada de casos de uso (Single Responsibility: orquestar, no ejecutar)."""

    def __init__(
        self,
        router: IntentRouter,
        registry: UserRegistry,
        cluster_factory: ClusterGatewayFactory,
        history: HistoryRepository,
        confirmations: ConfirmationStore,
        structures: StructureBuilder,
        job_tracker: JobTracker,
        calc_inputs: Optional[CalcInputGenerator] = None,
        potcar_dir: str = "",
        remote_base: str = "becario_runs",
    ) -> None:
        self._router = router
        self._registry = registry
        self._cluster_factory = cluster_factory
        self._history = history
        self._confirmations = confirmations
        self._structures = structures
        self._job_tracker = job_tracker
        self._calc_inputs = calc_inputs
        self._potcar_dir = potcar_dir.rstrip("/")
        self._remote_base = remote_base.rstrip("/")

    # ------------------------------------------------------------------
    # Entrada principal: texto del usuario
    # ------------------------------------------------------------------
    def handle_text(self, chat_id: int, user_id: int, text: str) -> Reply:
        identity = self._registry.get_identity(user_id)
        if identity is None:
            logger.warning("Mensaje de usuario no registrado: telegram_user_id=%s", user_id)
            return Reply(text=NOT_REGISTERED_TEXT)

        ctx = _Ctx(
            chat_id=chat_id,
            user_id=user_id,
            identity=identity,
            cluster=self._cluster_factory.for_identity(identity),
        )

        routed = self._router.route(text)
        logger.info(
            "routed intent=%s params=%s ssh_user=%s",
            routed.intent, routed.params, identity.ssh_user,
        )

        handlers = {
            Intent.SUBMIT_SLURM: self._prepare_submit,
            Intent.CANCEL_JOB: self._prepare_cancel,
            Intent.CHECK_STATUS: self._check_status,
            Intent.QUERY_DB: self._query_history,
            Intent.MODIFY_STRUCTURE: self._modify_structure,
            Intent.PREPARE_CALC: self._prepare_calc,
        }
        handler = handlers.get(routed.intent)
        if handler is None:
            return Reply(text=HELP_TEXT)
        return handler(ctx, routed.params)

    # ------------------------------------------------------------------
    # Confirmaciones
    # ------------------------------------------------------------------
    def confirm(self, token: str, requester_id: int) -> Reply:
        action = self._confirmations.peek(token)
        if action is None:
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)
        if action.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)

        identity = self._registry.get_identity(requester_id)
        if identity is None:  # se dio de baja entre el pedido y la confirmación
            self._confirmations.pop(token)
            return Reply(text=NOT_REGISTERED_TEXT)
        cluster = self._cluster_factory.for_identity(identity)

        action = self._confirmations.pop(token)  # recién ahora se consume
        if action is None:
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)

        if action.intent is Intent.SUBMIT_SLURM:
            # `workflow` viaja junto a los campos del job pero no es parte de
            # SlurmJobRequest (Pydantic ignora las claves extra del payload).
            req = SlurmJobRequest(**action.payload)
            result = cluster.submit_job(req)
            if result.ok and result.job_id:
                self._job_tracker.track(TrackedJob(
                    job_id=result.job_id,
                    owner_id=requester_id,
                    chat_id=action.chat_id,
                    ssh_user=identity.ssh_user,
                    job_name=req.job_name,
                    script_path=req.script_path,
                    workflow=action.payload.get("workflow", ""),
                ))
                self._history.add(
                    owner_id=requester_id, job_id=result.job_id,
                    nombre_trabajo=req.job_name, estado="enviado",
                )
                text = (
                    f"✅ 🚀 Trabajo enviado al cluster.\n"
                    f"• Job: {result.job_id}\n"
                    f"• Nombre: {req.job_name}\n"
                    f"• Partición: {req.partition if req.partition != 'default' else 'la default del cluster'}\n"
                    f"📌 Seguimiento activado para el job {result.job_id}: te aviso cuando termine."
                )
            elif result.ok:
                # sbatch aceptó el trabajo pero no pudimos leer el número de job.
                text = (
                    f"✅ 🚀 {result.message}\n"
                    "⚠️ No pude leer el número de job, así que no voy a poder avisarte "
                    "cuando termine. Consultalo con «estado de mis trabajos»."
                )
            else:
                text = f"❌ 🚀 {result.message}"
            return Reply(text=text)
        elif action.intent is Intent.CANCEL_JOB:
            jid = JobId(value=action.payload["job_id"])
            result = cluster.cancel_job(jid)
            if result.ok:
                # Ya lo sabe (lo canceló él mismo): evita el aviso duplicado
                # del monitor cuando vea el estado CANCELLED más tarde. Como el
                # monitor tampoco lo va a registrar, queda asentado acá.
                self._job_tracker.mark_notified(jid.value, requester_id)
                self._history.add(
                    owner_id=requester_id, job_id=jid.value,
                    nombre_trabajo="", estado="cancelado",
                )
            status = "✅" if result.ok else "❌"
            return Reply(text=f"{status} 🛑 {result.message}")
        else:  # pragma: no cover - defensivo
            return Reply(text="⚠️ Acción pendiente desconocida.")

    def reject(self, token: str, requester_id: int) -> Reply:
        action = self._confirmations.peek(token)
        if action is None:
            return Reply(text=EXPIRED_CONFIRMATION_TEXT)
        if action.requester_id != requester_id:
            return Reply(text=FOREIGN_CONFIRMATION_TEXT)
        self._confirmations.pop(token)
        return Reply(text="❌ Operación cancelada.")

    # ------------------------------------------------------------------
    # Casos de uso destructivos: preparan, no ejecutan
    # ------------------------------------------------------------------
    def _prepare_submit(self, ctx: _Ctx, params: dict) -> Reply:
        try:
            req = SlurmJobRequest(
                job_name=params.get("nombre_trabajo") or "becario_job",
                partition=params.get("particion") or "default",
                nodes=int(params.get("nodos") or 1),
                time_limit=params.get("tiempo_limite") or "01:00:00",
                script_path=params.get("script_remoto") or "/ruta/al/calculo.sh",
            )
        except (ValidationError, ValueError) as exc:
            return Reply(text=f"⚠️ Parámetros inválidos para el envío:\n{exc}")

        action = PendingAction(
            chat_id=ctx.chat_id,
            requester_id=ctx.user_id,
            intent=Intent.SUBMIT_SLURM,
            description=f"🚀 Enviar job Slurm (cuenta {ctx.identity.ssh_user}):\n{req.describe()}",
            payload=req.model_dump(),
        )
        token = self._confirmations.put(action)
        return Reply(
            text=f"⚠️ ¿Confirmás esta acción?\n\n{action.description}",
            needs_confirmation=True,
            confirmation_token=token,
        )

    def _prepare_cancel(self, ctx: _Ctx, params: dict) -> Reply:
        raw = params.get("job_id")
        if not raw:
            return Reply(text="⚠️ Falta el job_id para cancelar. Decime cuál trabajo querés cancelar.")
        try:
            jid = JobId(value=str(raw))
        except (ValidationError, ValueError):
            return Reply(text=f"⚠️ job_id inválido: {raw!r}")

        action = PendingAction(
            chat_id=ctx.chat_id,
            requester_id=ctx.user_id,
            intent=Intent.CANCEL_JOB,
            description=f"🛑 Cancelar el trabajo {jid} (cuenta {ctx.identity.ssh_user})",
            payload={"job_id": jid.value},
        )
        token = self._confirmations.put(action)
        return Reply(
            text=f"⚠️ ¿Confirmás esta acción?\n\n{action.description}",
            needs_confirmation=True,
            confirmation_token=token,
        )

    # ------------------------------------------------------------------
    # Casos de uso de solo lectura / no destructivos: ejecutan directo
    # ------------------------------------------------------------------
    def _check_status(self, ctx: _Ctx, params: dict) -> Reply:
        jid: Optional[JobId] = None
        raw = params.get("job_id")
        if raw:
            try:
                jid = JobId(value=str(raw))
            except (ValidationError, ValueError):
                return Reply(text=f"⚠️ job_id inválido: {raw!r}")
        result = ctx.cluster.job_status(jid)
        return Reply(text=f"📊 Estado trabajos ({ctx.identity.ssh_user}):\n{result.message}")

    def _query_history(self, ctx: _Ctx, params: dict) -> Reply:
        try:
            flt = HistoryFilter(
                job_id=params.get("job_id"),
                name_contains=params.get("filtro_busqueda"),
                owner_id=ctx.user_id,  # nunca viene del LLM
            )
        except (ValidationError, ValueError) as exc:
            return Reply(text=f"⚠️ Filtro inválido:\n{exc}")
        rows = self._history.search(flt)
        if not rows:
            return Reply(text="📋 Historial:\nNo se encontraron registros.")
        return Reply(text="📋 Historial:\n" + _format_history_table(rows), monospace=True)

    def _modify_structure(self, ctx: _Ctx, params: dict) -> Reply:
        formula = params.get("formula") or params.get("formula_quimica")
        if not formula:
            return Reply(
                text="⚠️ Decime qué estructura querés (fórmula), p. ej.: "
                '"generá un POSCAR de Si diamond 2x2x2".'
            )
        try:
            kind = (
                StructureKind.MOLECULE
                if str(params.get("tipo_estructura", "")).lower() == "molecule"
                else StructureKind.BULK
            )
            fmt_raw = str(params.get("formato_salida", "vasp")).lower()
            fmt = OutputFormat(fmt_raw) if fmt_raw in OutputFormat._value2member_map_ else OutputFormat.VASP
            sc = params.get("supercelda") or [1, 1, 1]
            req = StructureRequest(
                formula=str(formula),
                kind=kind,
                crystal=params.get("red_cristalina"),
                lattice_a=params.get("parametro_red"),
                supercell=tuple(int(x) for x in sc),
                vacuum=params.get("vacio"),
                output_format=fmt,
                remote_dest_dir=params.get("destino_remoto"),
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return Reply(text=f"⚠️ Parámetros de estructura inválidos:\n{exc}")

        try:
            result = self._structures.build(req)
        except Exception as exc:  # StructureBuildError y afines
            return Reply(text=f"⚠️ No pude construir la estructura:\n{exc}")

        uploaded_note = ""
        if req.remote_dest_dir:
            remote_path = f"{req.remote_dest_dir.rstrip('/')}/{result.filename}"
            up = ctx.cluster.upload_file(result.local_path, remote_path)
            uploaded_note = (
                f"\n📤 {up.message}" if up.ok else f"\n⚠️ Falló la subida: {up.message}"
            )

        return Reply(text=f"🔬 Estructura generada:\n{result.describe()}{uploaded_note}")

    # ------------------------------------------------------------------
    # Cálculo VASP completo: genera inputs, sube y deja el sbatch listo
    # ------------------------------------------------------------------

    # Orden de búsqueda del pseudopotencial de cada elemento en la
    # biblioteca del cluster (variantes semi-core primero, como recomienda
    # VASP para la mayoría de los metales de transición).
    _POTCAR_VARIANTS = ("_sv", "_pv", "")

    def _prepare_calc(self, ctx: _Ctx, params: dict) -> Reply:
        if self._calc_inputs is None:
            return Reply(
                text="⚠️ La preparación de cálculos VASP no está configurada en este bot."
            )
        formula = params.get("formula") or params.get("formula_quimica")
        if not formula:
            return Reply(
                text="⚠️ Decime qué material querés calcular, p. ej.: "
                '"relajá los parámetros de red del bulk de W".'
            )
        if not self._potcar_dir or not self._potcar_dir.startswith("/"):
            return Reply(
                text="⚠️ Falta configurar la biblioteca de POTCAR del cluster "
                "(BECARIO_POTCAR_DIR, ruta absoluta) para preparar cálculos VASP."
            )

        kind_raw = str(params.get("tipo_calculo") or "").strip().lower()
        calc_kind = (
            CalcKind(kind_raw)
            if kind_raw in CalcKind._value2member_map_
            else CalcKind.STATIC
        )

        # Barrido de ENCUT explícito (si el usuario dio rango); si no, el
        # modelo de dominio usa su default.
        encut_values = None
        lo, hi = params.get("encut_min"), params.get("encut_max")
        if calc_kind is CalcKind.ENCUT_SCAN and lo and hi:
            step = int(params.get("encut_paso") or 50)
            if step <= 0:
                return Reply(text="⚠️ El paso del barrido de ENCUT debe ser positivo.")
            encut_values = list(range(int(lo), int(hi) + 1, step))

        try:
            sc = params.get("supercelda") or [1, 1, 1]
            kp = params.get("puntos_k")
            req = VaspCalcRequest(
                formula=str(formula),
                crystal=params.get("red_cristalina"),
                lattice_a=params.get("parametro_red"),
                supercell=tuple(int(x) for x in sc),
                calc_kind=calc_kind,
                encut=int(params.get("encut") or 520),
                kpoints=tuple(int(x) for x in kp) if kp else None,
                encut_values=encut_values,
                partition=params.get("particion") or "default",
                nodes=int(params.get("nodos") or 1),
                time_limit=params.get("tiempo_limite") or "01:00:00",
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return Reply(text=f"⚠️ Parámetros de cálculo inválidos:\n{exc}")

        try:
            result = self._calc_inputs.generate(req)
        except Exception as exc:  # StructureBuildError y afines
            return Reply(text=f"⚠️ No pude generar los inputs:\n{exc}")

        # POTCAR: una variante por elemento, en el orden de especies del POSCAR.
        potcar_sources: list[str] = []
        for element in result.elements:
            candidates = [
                f"{self._potcar_dir}/{element}{suffix}/POTCAR"
                for suffix in self._POTCAR_VARIANTS
            ]
            found = next((c for c in candidates if ctx.cluster.file_exists(c)), None)
            if found is None:
                tried = ", ".join(f"{element}{s}" for s in self._POTCAR_VARIANTS)
                return Reply(
                    text=f"⚠️ No encontré POTCAR para {element} en "
                    f"{self._potcar_dir} (busqué {tried})."
                )
            potcar_sources.append(found)

        base = self._remote_base
        if not base.startswith("/"):
            home = ctx.cluster.home_dir()
            if not home:
                return Reply(
                    text="⚠️ No pude resolver el home remoto para ubicar la corrida. "
                    "Revisá la conexión al cluster."
                )
            base = f"{home}/{base}"
        remote_run_dir = f"{base}/{result.run_name}"

        up = ctx.cluster.upload_dir(result.local_dir, remote_run_dir)
        if not up.ok:
            return Reply(text=f"⚠️ Falló la subida de los inputs: {up.message}")

        cat = ctx.cluster.concat_files(potcar_sources, f"{remote_run_dir}/POTCAR")
        if not cat.ok:
            return Reply(text=f"⚠️ No pude armar el POTCAR en el cluster: {cat.message}")

        try:
            slurm_req = SlurmJobRequest(
                job_name=f"{req.formula}_{calc_kind.value}",
                partition=req.partition,
                nodes=req.nodes,
                time_limit=req.time_limit,
                script_path=f"{remote_run_dir}/run_vasp.sh",
            )
        except (ValidationError, ValueError) as exc:
            return Reply(text=f"⚠️ Parámetros inválidos para el envío:\n{exc}")

        workflow = "encut_scan" if calc_kind is CalcKind.ENCUT_SCAN else ""
        payload = slurm_req.model_dump()
        payload["workflow"] = workflow
        action = PendingAction(
            chat_id=ctx.chat_id,
            requester_id=ctx.user_id,
            intent=Intent.SUBMIT_SLURM,
            description=(
                f"🧪 Enviar cálculo VASP ({calc_kind.value}, cuenta "
                f"{ctx.identity.ssh_user}):\n{result.describe()}\n"
                f"📂 {remote_run_dir}\n{slurm_req.describe()}"
            ),
            payload=payload,
        )
        token = self._confirmations.put(action)
        return Reply(
            text=(
                f"✅ Inputs generados y subidos al cluster.\n\n"
                f"⚠️ ¿Confirmás el envío?\n\n{action.description}"
            ),
            needs_confirmation=True,
            confirmation_token=token,
        )
