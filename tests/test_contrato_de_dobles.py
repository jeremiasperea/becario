"""Los dobles tienen que devolver lo MISMO que el puerto que sustituyen.

Existe por un bug concreto, y por el hecho de que ese bug es el tercero de
la misma familia:

- `chat_id` viajaba como objeto `Chat` y no como entero. Los diez tests de
  `start_modification` pasaban `chat_id=1`; el aviso moría después, en el
  `send_message` real (D4 de `plan_refactor_bot.md`).
- `Notification` creció con `acuse` y los tests del tick del monitor
  seguían armándola con un `SimpleNamespace`, así que el camino nuevo
  habría quedado sin cubrir.
- `ClusterGateway.job_state` pasó de `Optional[str]` a `JobStateReading`, y
  `relaxed_source` quedó pasándole el objeto entero a
  `JobStatus.from_slurm`: `AttributeError` en producción, con 1050 tests en
  verde y 28 escenarios de batería también. El doble de
  `test_relaxed_source` devolvía un `str` con forma propia.

El patrón es siempre el mismo: **un doble que se parece lo suficiente para
pasar y no lo suficiente para servir.** Los tres se detectan igual —
comparando el doble contra el puerto en vez de contra sí mismo — y eso es
lo que hace este módulo. Cuando un puerto cambie de tipo, esto falla antes
de que el bug llegue a producción.
"""
from typing import Optional

import pytest

from becario.domain.models import JobId, JobStateReading
from becario.domain.ports import CalcRunRepository

from .test_job_monitor import FakeCluster as ClusterDelMonitor
from .test_relaxed_source import FakeCalcRuns as CorridasDeRelajadas
from .test_relaxed_source import FakeCluster as ClusterDeRelajadas
from .test_service import FakeCalcRuns as CorridasDelServicio
from .test_service import FakeCluster as ClusterDelServicio


def _dobles() -> list:
    """Un ejemplar de cada doble de `ClusterGateway` de la suite.

    Se instancian a mano y no por descubrimiento automático: son tres, y
    una lista explícita se lee mejor que la magia. Si aparece un cuarto y
    nadie lo agrega acá, el que lo escriba se entera cuando su puerto
    cambie — que es exactamente el día en que importa.
    """
    return [
        ClusterDelServicio(ssh_user="alice"),
        ClusterDelMonitor(state="RUNNING"),
        ClusterDeRelajadas(),
    ]


@pytest.mark.parametrize("doble", _dobles(), ids=lambda d: type(d).__module__.split(".")[-1])
class TestLosDoblesRespetanElPuerto:
    def test_job_state_devuelve_una_lectura_no_un_string(self, doble):
        """El caso que se pagó. Un `str` acá pasa todos los tests del
        módulo que lo define y revienta en el primer llamador real."""
        lectura = doble.job_state(JobId(value="42"))

        assert isinstance(lectura, JobStateReading), (
            f"{type(doble).__module__}.{type(doble).__name__}.job_state devolvió "
            f"{type(lectura).__name__}; el puerto promete JobStateReading"
        )
        assert isinstance(lectura.reachable, bool)
        assert lectura.state is None or isinstance(lectura.state, str)

    def test_lo_que_devuelve_sirve_para_lo_que_lo_usan(self, doble):
        """No alcanza con el tipo: `JobStatus.from_slurm` recibe el `.state`,
        y ese contrato es el que se rompió."""
        from becario.domain.models import JobStatus

        lectura = doble.job_state(JobId(value="42"))
        assert isinstance(JobStatus.from_slurm(lectura.state), JobStatus)


def _dobles_de_corridas() -> list:
    """Un ejemplar de cada doble de `CalcRunRepository` de la suite.

    Se suma acá porque el puerto acaba de crecer —`find_by_job_id`, para
    poder anclar en «la corrida del job N»— y esa es exactamente la
    situación que este módulo existe para vigilar: un método nuevo en el
    puerto y dos dobles que siguen sin tenerlo pasan todos sus tests, y
    revientan con `AttributeError` en el primer llamador real.
    """
    return [CorridasDelServicio(), CorridasDeRelajadas()]


@pytest.mark.parametrize(
    "doble", _dobles_de_corridas(), ids=lambda d: type(d).__module__.split(".")[-1]
)
class TestLosDoblesDeCorridasRespetanElPuerto:
    def test_tienen_todos_los_metodos_del_puerto(self, doble):
        faltan = [
            nombre for nombre in vars(CalcRunRepository)
            if not nombre.startswith("_") and not callable(getattr(doble, nombre, None))
        ]

        assert faltan == [], (
            f"{type(doble).__module__}.{type(doble).__name__} no implementa "
            f"{faltan}; el puerto CalcRunRepository sí"
        )

    def test_buscar_por_job_devuelve_un_registro_o_nada(self, doble):
        """El puerto promete `Optional[dict]`, no una lista. Un doble que
        devolviera `[]` pasa un `if row is None` y se rompe en el `.get`."""
        encontrado = doble.find_by_job_id(0, "no-existe")

        assert encontrado is None or isinstance(encontrado, dict)
