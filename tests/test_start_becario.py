"""Tests del launcher `scripts/start_becario.py`.

Todo corre sin red y sin subprocesos reales: `probe` se monkeypatchea y el
`OllamaRouter` nunca llega a hacer HTTP. Lo que se valida es la MÁQUINA DE
DECISIÓN — cuándo levantar el servidor, cuándo negarse porque es remoto,
cuándo bajar el modelo — que es donde este script se puede equivocar feo.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.infrastructure.ollama_router import (
    OllamaModelMissingError,
    OllamaUnreachableError,
)
from scripts import start_becario


class TestIsLocalUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://0.0.0.0:11434",
            "http://[::1]:11434",
        ],
    )
    def test_local_hosts(self, url):
        assert start_becario.is_local_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://192.168.1.50:11434",
            "http://gpu-server.lan:11434",
            "https://ollama.example.com",
        ],
    )
    def test_remote_hosts(self, url):
        assert start_becario.is_local_url(url) is False


class TestOllamaHostEnv:
    def test_default_port_when_absent(self):
        assert start_becario.ollama_host_env("http://localhost") == "localhost:11434"

    def test_respects_custom_port(self):
        assert start_becario.ollama_host_env("http://localhost:11500") == "localhost:11500"


class TestAsk:
    @pytest.mark.parametrize("respuesta", ["s", "si", "sí", "y", "YES", " S "])
    def test_afirmativas(self, monkeypatch, respuesta):
        monkeypatch.setattr("builtins.input", lambda _: respuesta)

        assert start_becario._ask("¿?") is True

    @pytest.mark.parametrize("respuesta", ["n", "no", "", "cualquiera"])
    def test_negativas(self, monkeypatch, respuesta):
        monkeypatch.setattr("builtins.input", lambda _: respuesta)

        assert start_becario._ask("¿?") is False

    def test_sin_terminal_es_no(self, monkeypatch):
        """Sin TTY (systemd, CI, pipe) `input` tira EOFError: eso NO es un sí."""

        def boom(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", boom)

        assert start_becario._ask("¿?") is False


class _FakeRouter:
    """Router que levanta lo que se le indique en `ensure_model_available`."""

    def __init__(self, error=None):
        self.error = error

    def ensure_model_available(self, *, timeout=10.0):
        if self.error is not None:
            raise self.error


class TestProbe:
    def test_ok_when_no_error(self):
        assert start_becario.probe(_FakeRouter()) == "ok"

    def test_caido_when_unreachable(self):
        router = _FakeRouter(OllamaUnreachableError("sin conexión"))
        assert start_becario.probe(router) == "caido"

    def test_sin_modelo_when_missing(self):
        router = _FakeRouter(OllamaModelMissingError("gemma4:12b", available=[]))
        assert start_becario.probe(router) == "sin-modelo"


class _ProbeSequence:
    """Devuelve estados sucesivos, repitiendo el último."""

    def __init__(self, *states):
        self.states = list(states)
        self.calls = 0

    def __call__(self, router, *, timeout=2.0):
        self.calls += 1
        index = min(self.calls - 1, len(self.states) - 1)
        return self.states[index]


class TestEnsureOllama:
    def test_no_hace_nada_si_ya_esta_ok(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("ok"))
        monkeypatch.setattr(
            start_becario, "start_ollama", lambda *a, **k: pytest.fail("no debía arrancar")
        )
        monkeypatch.setattr(
            start_becario, "pull_model", lambda *a, **k: pytest.fail("no debía bajar nada")
        )

        start_becario.ensure_ollama(
            "http://localhost:11434", "gemma3:4b", timeout=1.0, assume_yes=True
        )

    def test_arranca_el_server_si_esta_caido_y_es_local(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("caido", "ok"))
        arrancados = []
        monkeypatch.setattr(
            start_becario, "start_ollama", lambda url, router, **k: arrancados.append(url)
        )

        start_becario.ensure_ollama(
            "http://localhost:11434", "gemma3:4b", timeout=1.0, assume_yes=True
        )

        assert arrancados == ["http://localhost:11434"]

    def test_no_intenta_arrancar_un_server_remoto(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("caido"))
        monkeypatch.setattr(
            start_becario, "start_ollama", lambda *a, **k: pytest.fail("es remoto")
        )

        with pytest.raises(SystemExit) as exc:
            start_becario.ensure_ollama(
                "http://gpu-server.lan:11434", "gemma3:4b", timeout=1.0, assume_yes=True
            )

        assert exc.value.code == 1

    def test_baja_el_modelo_si_falta_y_es_local(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("sin-modelo"))
        bajados = []
        monkeypatch.setattr(
            start_becario, "pull_model", lambda model, **k: bajados.append(model)
        )

        start_becario.ensure_ollama(
            "http://localhost:11434", "gemma3:4b", timeout=1.0, assume_yes=True
        )

        assert bajados == ["gemma3:4b"]

    def test_no_baja_modelos_en_un_server_remoto(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("sin-modelo"))
        monkeypatch.setattr(
            start_becario, "pull_model", lambda *a, **k: pytest.fail("es remoto")
        )

        with pytest.raises(SystemExit) as exc:
            start_becario.ensure_ollama(
                "http://gpu-server.lan:11434", "gemma3:4b", timeout=1.0, assume_yes=True
            )

        assert exc.value.code == 1

    def test_baja_el_modelo_si_falta_despues_de_arrancar(self, monkeypatch):
        """Server caído -> arranca -> recién ahí se ve que falta el modelo."""
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("caido", "sin-modelo"))
        monkeypatch.setattr(start_becario, "start_ollama", lambda *a, **k: None)
        bajados = []
        monkeypatch.setattr(
            start_becario, "pull_model", lambda model, **k: bajados.append(model)
        )

        start_becario.ensure_ollama(
            "http://localhost:11434", "gemma3:4b", timeout=1.0, assume_yes=True
        )

        assert bajados == ["gemma3:4b"]


class _Completed:
    def __init__(self, returncode):
        self.returncode = returncode


class TestPullModel:
    def test_aborta_si_el_usuario_dice_que_no(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with pytest.raises(SystemExit) as exc:
            start_becario.pull_model("gemma3:4b", assume_yes=False)

        assert exc.value.code == 1

    def test_no_pregunta_con_assume_yes(self, monkeypatch):
        monkeypatch.setattr(
            "builtins.input", lambda _: pytest.fail("no debía preguntar")
        )
        monkeypatch.setattr(
            start_becario.subprocess, "run", lambda *a, **k: _Completed(0)
        )

        start_becario.pull_model("gemma3:4b", assume_yes=True)

    def test_aborta_si_el_pull_falla(self, monkeypatch):
        monkeypatch.setattr(
            start_becario.subprocess, "run", lambda *a, **k: _Completed(1)
        )

        with pytest.raises(SystemExit) as exc:
            start_becario.pull_model("gemma3:4b", assume_yes=True)

        assert exc.value.code == 1


class _FakeProcess:
    """Popen falso: `alive` controla si sigue vivo en el poll."""

    def __init__(self, alive=True):
        self.alive = alive
        self.pid = 1234

    def poll(self):
        return None if self.alive else 1


class TestWaitUntilReachable:
    def test_true_cuando_el_server_responde(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("caido", "ok"))
        import time

        assert start_becario.wait_until_reachable(
            _FakeRouter(), _FakeProcess(), deadline=time.monotonic() + 5
        )

    def test_false_si_el_proceso_muere(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("caido"))
        import time

        assert not start_becario.wait_until_reachable(
            _FakeRouter(), _FakeProcess(alive=False), deadline=time.monotonic() + 5
        )

    def test_false_al_vencer_el_deadline(self, monkeypatch):
        monkeypatch.setattr(start_becario, "probe", _ProbeSequence("caido"))
        import time

        assert not start_becario.wait_until_reachable(
            _FakeRouter(), _FakeProcess(), deadline=time.monotonic() - 1
        )
