"""Tests de la validación de arranque de Ollama (servidor + modelo).

Cubre `OllamaRouter._model_matches` (regla de matching, ADR-3) y
`OllamaRouter.ensure_model_available` (chequeo de reachability + presencia
de modelo, `GET /api/tags`), ambos parte del fail-fast de arranque descripto
en la spec `sdd/validate-ollama-model-on-startup`.
"""
import httpx
import pytest

from becario.infrastructure.ollama_router import (
    OllamaModelMissingError,
    OllamaRouter,
    OllamaUnreachableError,
    OllamaValidationError,
)


class TestModelMatches:
    """`_model_matches` es un helper puro: sin mocks, tabla de casos."""

    def test_exact_name_tag_match(self):
        assert OllamaRouter._model_matches(
            "gemma4:12b", ["gemma3:4b", "gemma4:12b"]
        )

    def test_exact_name_tag_mismatch_different_tag(self):
        # Un tag distinto configurado explícitamente NUNCA matchea otro tag
        # del mismo repo (evita falsos positivos, ver ADR-3).
        assert not OllamaRouter._model_matches("gemma4:12b", ["gemma4:2b"])

    def test_bare_name_resolves_to_latest_present(self):
        assert OllamaRouter._model_matches("llama3", ["llama3:latest"])

    def test_bare_name_absent_when_only_other_tag_present(self):
        # Sin ":latest" en el server, el nombre "pelado" no matchea otro tag.
        assert not OllamaRouter._model_matches("llama3", ["llama3:8b"])

    def test_bare_name_does_not_match_unrelated_model(self):
        assert not OllamaRouter._model_matches("gemma4", ["gemma3:4b"])


class TestEnsureModelAvailable:
    """`ensure_model_available` mockea `httpx.get` (adaptador, no red real)."""

    def _router(self) -> OllamaRouter:
        return OllamaRouter(base_url="http://localhost:11434", model="gemma4:12b")

    def test_model_present_exact_match_returns_none(self, monkeypatch):
        response = httpx.Response(
            200,
            json={"models": [{"name": "gemma4:12b"}, {"name": "gemma3:4b"}]},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **k: response)
        assert self._router().ensure_model_available() is None

    def test_bare_configured_model_matches_latest_returns_none(self, monkeypatch):
        response = httpx.Response(
            200,
            json={"models": [{"name": "gemma4:latest"}]},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **k: response)
        router = OllamaRouter(base_url="http://localhost:11434", model="gemma4")
        assert router.ensure_model_available() is None

    def test_model_missing_raises_with_available_populated(self, monkeypatch):
        response = httpx.Response(
            200,
            json={"models": [{"name": "gemma3:4b"}, {"name": "llama3:8b"}]},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **k: response)
        with pytest.raises(OllamaModelMissingError) as exc_info:
            self._router().ensure_model_available()
        assert exc_info.value.model == "gemma4:12b"
        assert exc_info.value.available == ["gemma3:4b", "llama3:8b"]

    def test_transport_error_raises_unreachable(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", _raise)
        with pytest.raises(OllamaUnreachableError):
            self._router().ensure_model_available()

    def test_malformed_json_body_raises_unreachable(self, monkeypatch):
        response = httpx.Response(
            200,
            content=b"esto no es json",
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **k: response)
        with pytest.raises(OllamaUnreachableError):
            self._router().ensure_model_available()

    def test_missing_models_key_raises_unreachable(self, monkeypatch):
        response = httpx.Response(
            200,
            json={"unexpected": []},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **k: response)
        with pytest.raises(OllamaUnreachableError):
            self._router().ensure_model_available()

    def test_both_exceptions_share_common_base(self):
        assert issubclass(OllamaUnreachableError, OllamaValidationError)
        assert issubclass(OllamaModelMissingError, OllamaValidationError)
