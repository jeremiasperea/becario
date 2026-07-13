"""Tests de `_check_ollama_model`, el chequeo de arranque que valida
reachability + presencia del modelo de Ollama antes de arrancar el bot
(ver spec `sdd/validate-ollama-model-on-startup`, Startup Integration)."""
import pytest

import main
from becario.infrastructure.ollama_router import (
    OllamaModelMissingError,
    OllamaRouter,
    OllamaUnreachableError,
)


def _settings(**overrides):
    defaults = dict(
        telegram_token="dummy-token",
        ssh_host="cluster.example.com",
        ollama_url="http://localhost:11434",
        ollama_model="gemma4:12b",
    )
    defaults.update(overrides)
    return main.Settings(**defaults)


class TestCheckOllamaModel:
    def test_unreachable_server_exits(self, monkeypatch):
        def _raise(self, *, timeout=10.0):
            raise OllamaUnreachableError("no se pudo consultar el servidor")

        monkeypatch.setattr(OllamaRouter, "ensure_model_available", _raise)
        with pytest.raises(SystemExit):
            main._check_ollama_model(_settings())

    def test_missing_model_exits(self, monkeypatch):
        def _raise(self, *, timeout=10.0):
            raise OllamaModelMissingError("gemma4:12b", available=["gemma3:4b"])

        monkeypatch.setattr(OllamaRouter, "ensure_model_available", _raise)
        with pytest.raises(SystemExit):
            main._check_ollama_model(_settings())

    def test_model_present_returns_without_exit(self, monkeypatch):
        monkeypatch.setattr(
            OllamaRouter, "ensure_model_available", lambda self, *, timeout=10.0: None
        )
        assert main._check_ollama_model(_settings()) is None
