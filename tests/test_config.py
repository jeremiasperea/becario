"""Tests de carga de configuración: loader de .env y Settings.from_env.

Se verifica que un usuario pueda tener sus secretos en un archivo local y que
el entorno siempre gane sobre ese archivo. Todos los tests usan tmp_path y
monkeypatch para no depender de un .env real del desarrollador.
"""
import pytest

from becario import config
from becario.config import ConfigError, Settings, load_local_env, required_config_present


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadLocalEnv:
    def test_parsea_key_value(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BECARIO_SSH_HOST", raising=False)
        env = _write(tmp_path / ".env", "BECARIO_SSH_HOST=cluster.test.edu\n")
        load_local_env(env)
        assert config.os.environ["BECARIO_SSH_HOST"] == "cluster.test.edu"

    def test_ignora_comentarios_y_blancos(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BECARIO_SSH_HOST", raising=False)
        env = _write(
            tmp_path / ".env",
            "# comentario\n\n   \nBECARIO_SSH_HOST=h1\n",
        )
        load_local_env(env)
        assert config.os.environ["BECARIO_SSH_HOST"] == "h1"

    def test_soporta_export_y_comillas(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BECARIO_BOT_TOKEN", raising=False)
        monkeypatch.delenv("BECARIO_SSH_HOST", raising=False)
        env = _write(
            tmp_path / ".env",
            'export BECARIO_BOT_TOKEN="123:abc"\n' "BECARIO_SSH_HOST='h2'\n",
        )
        load_local_env(env)
        assert config.os.environ["BECARIO_BOT_TOKEN"] == "123:abc"
        assert config.os.environ["BECARIO_SSH_HOST"] == "h2"

    def test_no_pisa_variables_ya_seteadas(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BECARIO_SSH_HOST", "del-entorno")
        env = _write(tmp_path / ".env", "BECARIO_SSH_HOST=del-archivo\n")
        load_local_env(env)
        assert config.os.environ["BECARIO_SSH_HOST"] == "del-entorno"

    def test_archivo_ausente_no_falla(self, tmp_path):
        load_local_env(tmp_path / "no-existe.env")  # no debe lanzar


class TestRequiredConfigPresent:
    def test_true_cuando_estan_las_obligatorias(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BECARIO_BOT_TOKEN", raising=False)
        monkeypatch.delenv("BECARIO_SSH_HOST", raising=False)
        env = _write(
            tmp_path / ".env",
            "BECARIO_BOT_TOKEN=123:abc\nBECARIO_SSH_HOST=h\n",
        )
        assert required_config_present(env) is True

    def test_false_cuando_falta_alguna(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BECARIO_BOT_TOKEN", raising=False)
        monkeypatch.delenv("BECARIO_SSH_HOST", raising=False)
        env = _write(tmp_path / ".env", "BECARIO_SSH_HOST=h\n")
        assert required_config_present(env) is False


class TestSettingsFromEnv:
    def _clear(self, monkeypatch):
        for name in (
            "BECARIO_BOT_TOKEN", "BECARIO_SSH_HOST", "BECARIO_SSH_PORT",
            "BECARIO_OLLAMA_URL", "BECARIO_OLLAMA_MODEL", "BECARIO_CONFIRM_TTL",
            "BECARIO_MONITOR_INTERVAL",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_lee_desde_env_file(self, tmp_path, monkeypatch):
        self._clear(monkeypatch)
        env = _write(
            tmp_path / ".env",
            "BECARIO_BOT_TOKEN=123:abc\n"
            "BECARIO_SSH_HOST=cluster.test\n"
            "BECARIO_SSH_PORT=2222\n",
        )
        monkeypatch.setattr(config, "ENV_PATH", env)
        settings = Settings.from_env()
        assert settings.telegram_token == "123:abc"
        assert settings.ssh_host == "cluster.test"
        assert settings.ssh_port == 2222
        assert settings.ollama_model == "gemma4:12b"  # default

    def test_falta_token_lanza_config_error(self, tmp_path, monkeypatch):
        self._clear(monkeypatch)
        env = _write(tmp_path / ".env", "BECARIO_SSH_HOST=h\n")
        monkeypatch.setattr(config, "ENV_PATH", env)
        with pytest.raises(ConfigError):
            Settings.from_env()

    def test_puerto_no_numerico_lanza_config_error(self, tmp_path, monkeypatch):
        self._clear(monkeypatch)
        env = _write(
            tmp_path / ".env",
            "BECARIO_BOT_TOKEN=123:abc\nBECARIO_SSH_HOST=h\nBECARIO_SSH_PORT=abc\n",
        )
        monkeypatch.setattr(config, "ENV_PATH", env)
        with pytest.raises(ConfigError):
            Settings.from_env()
