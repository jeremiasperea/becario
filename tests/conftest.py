"""Configuración compartida de la suite.

Regla que se fija acá: **ningún test espera de verdad.** Con el backoff de
`becario/reintentos.py` en tres bordes (SSH, Ollama, Materials Project),
cada test que ejercita un fallo transitorio pagaba las pausas reales — se
midió en +3 segundos de suite el día que se agregaron, y eso crece solo con
cada test nuevo de reintento. Una suite lenta se corre menos.
"""
import pytest

from becario import reintentos


@pytest.fixture(autouse=True)
def sin_esperas_de_reintento(monkeypatch):
    """Neutraliza las pausas del backoff en toda la suite.

    Se parchea `reintentos.dormir_backoff` y NO `time.sleep`: el primer
    intento fue el segundo, y rompió siete tests que usan `time.sleep`
    legítimamente para hacer vencer un TTL. La costura tiene que apagar
    exactamente el backoff.

    Los adaptadores no exponen el parámetro `dormir` de `reintentar()`, ni
    tienen por qué: cuánto esperar entre reintentos es política del helper,
    no de quien lo usa. Un test que quiera VER las pausas le pasa su propio
    `dormir` a `reintentar()` y las registra en una lista.
    """
    monkeypatch.setattr(reintentos, "dormir_backoff", lambda _s: None)
