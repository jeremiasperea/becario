"""Ejecutor de planes multi-paso.

Generaliza el flujo de un solo paso a N pasos (ver ADR-0006): itera los
pasos EN ORDEN y corta en el primer fallo — sin rollback, lo ya
ejecutado queda hecho (mkdir/upload/sbatch no son transaccionales). El
reporte enumerado (✅ ejecutado / ❌ falló / ⏸ omitido) es lo que termina
viendo el usuario en la respuesta.

No es un puerto (Protocol) ni conoce `Intent`/`Plan`/`_Ctx`: solo sabe
correr una lista ordenada de "pasos" ya resueltos, cada uno una función
sin argumentos que devuelve `(ok, línea_de_reporte)`. Quien arma esa
lista (`BecarioService`, vía su `_step_executors()`) es quien conoce el
mapeo `Intent -> paso` — eso mantiene al ejecutor trivial de testear con
fakes puros, sin dobles de `ClusterGateway` ni de ningún puerto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Un paso resuelto: sin argumentos (todo lo que necesita ya quedó
# capturado en el closure al construirlo) y devuelve si salió bien más
# la línea de reporte correspondiente.
StepOutcome = tuple[bool, str]
StepFn = Callable[[], StepOutcome]


@dataclass(frozen=True)
class PlanExecutionResult:
    """Resultado de correr un plan completo.

    `ok` es `True` únicamente si TODOS los pasos se ejecutaron sin que
    ninguno fallara — un plan con un paso omitido (por un fallo previo)
    nunca es `ok`.
    """

    ok: bool
    report_lines: list[str] = field(default_factory=list)

    @property
    def report(self) -> str:
        return "\n".join(self.report_lines)


class PlanExecutor:
    """Corre una lista ordenada de pasos, cortando en el primer fallo."""

    def run(self, steps: list[StepFn]) -> PlanExecutionResult:
        lines: list[str] = []
        failed = False
        for i, step in enumerate(steps, start=1):
            if failed:
                lines.append(f"{i}. ⏸ omitido")
                continue
            ok, line = step()
            marker = "✅" if ok else "❌"
            lines.append(f"{i}. {marker} {line}")
            if not ok:
                failed = True
        return PlanExecutionResult(ok=not failed, report_lines=lines)
