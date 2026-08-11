"""Nada bloquea para siempre: deadline y keepalive SSH (T2), monitor
fuera del event loop (T3).

Parte de la batería de fallos inyectados de `docs/plan_tolerancia_fallos.md`.
El resto de la suite prueba que el sistema FUNCIONA; esta prueba que AGUANTA,
que es otra cosa: acá todos los dobles fallan a propósito. Ninguno de estos
tests necesita cluster, ni Ollama, ni red — y esa es la razón de que puedan
existir, y también de que no existieran: las fallas que cubren no se
reproducen pidiéndole cosas a un sistema sano.
"""
import asyncio
import logging
import paramiko
import threading
import time
from becario.application.context import Reply
from becario.infrastructure.ssh_gateway import SSHClusterGateway
from becario.presentation.telegram_bot import TelegramBot
from .test_telegram_bot import FAKE_TOKEN, FakeBotAPI, FakeChat, FakeService, FakeUpdate



# ---------------------------------------------------------------------------
# T2 — ningún comando SSH espera para siempre
# ---------------------------------------------------------------------------


class CanalColgado:
    """Un canal cuyo comando nunca termina: `status_event` no se setea
    jamás. Es lo que pasa de verdad cuando la conexión muere en silencio."""

    def __init__(self) -> None:
        self.status_event = threading.Event()  # nunca se setea
        self.cerrado = False

    def close(self) -> None:
        self.cerrado = True

    def recv_exit_status(self) -> int:  # pragma: no cover - no debe llamarse
        raise AssertionError("no se puede pedir el exit status de algo colgado")


class CanalNormal:
    def __init__(self, exit_code: int = 0) -> None:
        self.status_event = threading.Event()
        self.status_event.set()
        self._exit_code = exit_code
        self.cerrado = False

    def close(self) -> None:
        self.cerrado = True

    def recv_exit_status(self) -> int:
        return self._exit_code


class ChorroFalso:
    def __init__(self, channel, payload: bytes = b"") -> None:
        self.channel = channel
        self._payload = payload

    def read(self, *_a) -> bytes:
        return self._payload


class ClienteFalso:
    def __init__(self, channel) -> None:
        self._channel = channel

    def exec_command(self, command, timeout=None):
        return (
            None,
            ChorroFalso(self._channel, b"salida"),
            ChorroFalso(self._channel, b""),
        )


def _gateway(channel, command_timeout: float = 0.2) -> SSHClusterGateway:
    gw = SSHClusterGateway(
        host="h", user="u", key_path="/dev/null", command_timeout=command_timeout
    )
    gw._connection = lambda: ClienteFalso(channel)
    return gw


class TestComandoSSHConDeadline:
    def test_un_comando_colgado_corta_y_no_espera_para_siempre(self, caplog):
        """El test que no se podía escribir antes: `recv_exit_status()` es un
        `status_event.wait()` sin timeout, así que este caso colgaba el hilo
        —y con él una porción del bot— hasta reiniciar el proceso."""
        canal = CanalColgado()
        gw = _gateway(canal, command_timeout=0.2)

        arrancado = time.monotonic()
        with caplog.at_level(logging.ERROR):
            result = gw._run("sleep infinity")
        tardanza = time.monotonic() - arrancado

        assert result.ok is False
        assert "no respondió" in result.message
        assert tardanza < 5, "siguió esperando: el deadline no se aplicó"
        # El canal se cierra: si no, el hilo lector de paramiko lo sigue
        # sosteniendo y el descriptor queda colgado.
        assert canal.cerrado is True

    def test_un_comando_normal_no_cambia(self):
        gw = _gateway(CanalNormal(exit_code=0))
        result = gw._run("echo hola")
        assert result.ok is True
        assert result.stdout == "salida"

    def test_un_canal_que_muere_sin_codigo_cuenta_como_fallo(self):
        # `status_event` también se setea al cerrarse el canal; ahí paramiko
        # devuelve -1. Un comando que no pudo decir cómo terminó no terminó
        # bien.
        gw = _gateway(CanalNormal(exit_code=-1))
        assert gw._run("algo").ok is False


class TestKeepalive:
    def test_la_conexion_pide_keepalive(self, monkeypatch):
        """Sin esto, una conexión muerta en silencio no se detecta NUNCA y
        el deadline de arriba se convierte en el único freno."""
        pedidos = []

        class TransporteFalso:
            def set_keepalive(self, intervalo):
                pedidos.append(intervalo)

        class ClientePorConectar:
            def load_system_host_keys(self): ...
            def set_missing_host_key_policy(self, policy): ...
            def connect(self, **kwargs): ...
            def get_transport(self):
                return TransporteFalso()

        monkeypatch.setattr(paramiko, "SSHClient", ClientePorConectar)
        gw = SSHClusterGateway(
            host="h", user="u", key_path="/dev/null", keepalive_interval=30
        )
        gw._connection()

        assert pedidos == [30]


# ---------------------------------------------------------------------------
# T3 — el monitor no corre dentro del event loop
# ---------------------------------------------------------------------------


# Cuánto se queda el monitor adentro si nadie lo suelta. Es el margen que
# separa "corrió en un hilo" de "bloqueó el loop", así que tiene que ser
# holgadamente mayor que lo que tarda atender un mensaje.
_RETENCION = 3.0


class MonitorTrabado:
    """Se queda DENTRO de `poll_and_notify` hasta que el test lo suelte.

    Un `time.sleep()` no alcanza para probar esto, y me comí el error: con
    el monitor corriendo inline, el bloqueo ocurre mientras el test todavía
    está esperando a que el tick arranque, así que para cuando uno mira el
    reloj ya pasó — y el test daba verde con el bug puesto. Hace falta
    sincronización explícita: el test tiene que poder afirmar "el monitor
    está adentro AHORA" y recién entonces mandar un mensaje.
    """

    def __init__(self) -> None:
        self.adentro = threading.Event()
        self.soltar = threading.Event()
        self.hilos: list[str] = []

    def poll_and_notify(self):
        self.hilos.append(threading.current_thread().name)
        self.adentro.set()
        self.soltar.wait(_RETENCION)
        return []


class TestMonitorFueraDelLoop:
    def test_el_bot_contesta_con_el_monitor_trabado_adentro(self):
        """La prueba de que T3 está cerrado.

        Con el monitor sincrónico dentro del loop, el `soltar.wait()` de
        arriba congela el event loop entero: no hay forma de que `_on_text`
        corra hasta que se libere, y el escenario tarda `_RETENCION`. En un
        hilo aparte, atender es inmediato.
        """
        monitor = MonitorTrabado()
        bot = TelegramBot(
            token=FAKE_TOKEN,
            service=FakeService(Reply(text="listo")),
            job_monitor=monitor,
        )
        chat = FakeChat(chat_id=555)
        context = type("C", (), {"bot": FakeBotAPI()})()

        async def escenario():
            tick = asyncio.create_task(bot._on_monitor_tick(context))
            while not monitor.adentro.is_set():  # el monitor ya está trabado
                await asyncio.sleep(0.01)
            await bot._on_text(FakeUpdate(chat, user_id=7, text="hola"), None)
            monitor.soltar.set()
            await tick

        arrancado = time.monotonic()
        asyncio.run(escenario())
        tardanza = time.monotonic() - arrancado

        assert chat.sent == ["listo"]
        assert tardanza < _RETENCION / 2, (
            f"el monitor bloqueó el event loop ({tardanza:.2f}s): el mensaje "
            "no se pudo atender hasta que el cluster contestó"
        )

    def test_el_monitor_corre_en_el_pool_propio(self):
        monitor = MonitorTrabado()
        monitor.soltar.set()  # que no se quede esperando
        bot = TelegramBot(
            token=FAKE_TOKEN,
            service=FakeService(Reply(text="x")),
            job_monitor=monitor,
        )
        context = type("C", (), {"bot": FakeBotAPI()})()

        asyncio.run(bot._on_monitor_tick(context))

        assert monitor.hilos, "el monitor no corrió"
        assert all(h.startswith("becario-blocking") for h in monitor.hilos), monitor.hilos
