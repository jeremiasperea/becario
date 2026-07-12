"""Tests del `PlanExecutor`: itera pasos en orden, corta en el primer
fallo (sin rollback) y arma el reporte enumerado que ve el usuario.

Los "pasos" son simples callables `() -> (ok, line)` — el ejecutor no
conoce `Intent` ni el dominio de BECARIO, solo esa forma. Esto permite
testearlo con fakes locales, sin pasar por `BecarioService`.
"""
from becario.application.plan_executor import PlanExecutor, PlanExecutionResult

# ---------------------------------------------------------------------------
# Fakes locales: un "job tracker" mínimo para probar el invariante
# "a lo sumo un trabajo rastreado" cuando el paso destructivo está al final.
# ---------------------------------------------------------------------------


class _FakeJobTracker:
    def __init__(self):
        self.tracked: list[str] = []

    def track(self, job_id: str) -> None:
        self.tracked.append(job_id)


def _ok_step(calls: list[str], name: str, line: str):
    def _step():
        calls.append(name)
        return True, line

    return _step


def _fail_step(calls: list[str], name: str, line: str):
    def _step():
        calls.append(name)
        return False, line

    return _step


def _submit_step(calls: list[str], tracker: _FakeJobTracker, job_id: str):
    def _step():
        calls.append("submit")
        tracker.track(job_id)
        return True, f"trabajo {job_id} enviado"

    return _step


class TestPlanExecutorHappyPath:
    def test_all_steps_execute_in_order_and_report_success(self):
        calls: list[str] = []
        steps = [
            _ok_step(calls, "a", "carpeta /run creada"),
            _ok_step(calls, "b", "listado de /data"),
        ]
        result = PlanExecutor().run(steps)
        assert isinstance(result, PlanExecutionResult)
        assert result.ok is True
        assert calls == ["a", "b"]  # ambos pasos corrieron, en orden
        assert result.report_lines == [
            "1. ✅ carpeta /run creada",
            "2. ✅ listado de /data",
        ]

    def test_single_step_plan_reports_one_line(self):
        calls: list[str] = []
        steps = [_ok_step(calls, "a", "estado consultado")]
        result = PlanExecutor().run(steps)
        assert result.ok is True
        assert result.report_lines == ["1. ✅ estado consultado"]


class TestPlanExecutorStopOnFailure:
    def test_stop_on_first_failure_omits_remaining_and_does_not_run_them(self):
        calls: list[str] = []
        steps = [
            _ok_step(calls, "a", "carpeta /run creada"),
            _fail_step(calls, "b", "ruta inválida: '/tmp/../etc'"),
            _ok_step(calls, "c", "nunca debería ejecutarse"),
        ]
        result = PlanExecutor().run(steps)

        assert result.ok is False
        # El paso 3 NUNCA se invocó: no hay ejecución más allá del fallo.
        assert calls == ["a", "b"]
        assert result.report_lines == [
            "1. ✅ carpeta /run creada",
            "2. ❌ ruta inválida: '/tmp/../etc'",
            "3. ⏸ omitido",
        ]

    def test_failure_on_first_step_omits_every_later_step(self):
        calls: list[str] = []
        steps = [
            _fail_step(calls, "a", "fórmula inválida"),
            _ok_step(calls, "b", "no debería correr"),
            _ok_step(calls, "c", "tampoco debería correr"),
        ]
        result = PlanExecutor().run(steps)

        assert result.ok is False
        assert calls == ["a"]
        assert result.report_lines == [
            "1. ❌ fórmula inválida",
            "2. ⏸ omitido",
            "3. ⏸ omitido",
        ]


class TestPlanExecutorDestructiveTailTracking:
    """El paso destructivo (p. ej. enviar_slurm) va siempre al final del
    plan (invariante ya validado por `Plan`, ver domain/models.py). Acá
    se prueba que el ejecutor respeta ese orden: si un paso previo falla,
    el destructivo (que rastrearía un job) nunca se alcanza."""

    def test_destructive_tail_runs_and_tracks_exactly_one_job_on_success(self):
        calls: list[str] = []
        tracker = _FakeJobTracker()
        steps = [
            _ok_step(calls, "prep", "inputs subidos"),
            _submit_step(calls, tracker, "4242"),
        ]
        result = PlanExecutor().run(steps)

        assert result.ok is True
        assert tracker.tracked == ["4242"]  # a lo sumo un trabajo rastreado
        assert calls == ["prep", "submit"]

    def test_destructive_tail_never_runs_when_an_earlier_step_fails(self):
        calls: list[str] = []
        tracker = _FakeJobTracker()
        steps = [
            _fail_step(calls, "prep", "no encontré POTCAR para Si"),
            _submit_step(calls, tracker, "4242"),
        ]
        result = PlanExecutor().run(steps)

        assert result.ok is False
        assert tracker.tracked == []  # el destructivo nunca se alcanzó
        assert "submit" not in calls
        assert result.report_lines == [
            "1. ❌ no encontré POTCAR para Si",
            "2. ⏸ omitido",
        ]
