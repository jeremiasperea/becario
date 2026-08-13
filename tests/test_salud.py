"""Tests de `scripts/salud.py`, el chequeo que contesta «¿está sano?».

Sin red y sin cluster: las sondas se ejercitan aparte y acá se prueba lo
que decide — el diagnóstico, los percentiles y la lectura de la base. Es
justo donde estaba el bug que apareció al correrlo por primera vez contra
la base de producción.
"""
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.salud import _diagnostico, _metricas, _pct  # noqa: E402


class _Settings:
    """Lo mínimo que `_metricas` mira de la configuración."""

    db_path = ""
    ollama_timeout_seconds = 180.0


def _base(tmp_path: Path, *, con_poll_attempts: bool = True) -> Path:
    """Una base como la que hay en producción. `con_poll_attempts=False`
    reproduce una anterior a la migración de T6."""
    db = tmp_path / "becario.db"
    ahora = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db) as c:
        c.execute(
            "CREATE TABLE decisiones_router (latency_seconds REAL, outcome TEXT, created_at TEXT)"
        )
        c.executemany(
            "INSERT INTO decisiones_router VALUES (?, ?, ?)",
            [(3.4, "routed", ahora), (30.0, "routed", ahora),
             (118.1, "error", ahora), (12.0, "confirmed", ahora)],
        )
        columnas = "job_id TEXT, notified INTEGER"
        if con_poll_attempts:
            columnas += ", poll_attempts INTEGER DEFAULT 0"
        c.execute(f"CREATE TABLE trabajos_monitoreados ({columnas})")
        valores = ("1", 0, 3) if con_poll_attempts else ("1", 0)
        c.execute(
            f"INSERT INTO trabajos_monitoreados VALUES ({','.join('?' * len(valores))})",
            valores,
        )
    return db


class TestPercentil:
    def test_el_p90_no_inventa_un_valor_que_no_se_midio(self):
        # Método del más cercano: devuelve una muestra real, no una
        # interpolación. Con decenas de muestras da lo mismo, y una
        # latencia inventada en un informe de salud confunde.
        muestras = [1.0, 2.0, 3.0, 100.0]
        assert _pct(muestras, 0.90) in muestras

    def test_sin_muestras_no_revienta(self):
        assert _pct([], 0.90) == 0.0

    def test_el_maximo_es_el_p100(self):
        assert _pct([5.0, 1.0, 9.0], 1.0) == 9.0


class TestLecturaDeLaBase:
    def test_resume_el_router_desde_lo_que_ya_se_registraba(self, tmp_path):
        s = _Settings()
        m = _metricas(_base(tmp_path), s, dias=7)

        assert m["router"]["decisiones"] == 4
        assert m["router"]["max"] == 118.1
        assert m["router"]["desenlaces"] == {"routed": 2, "error": 1, "confirmed": 1}

    def test_una_base_anterior_a_la_migracion_no_lo_rompe(self, tmp_path):
        """El bug real: `poll_attempts` la agrega la migración suave al
        instanciar el tracker, y este script abre la base en SOLO LECTURA
        sin pasar por él. Contra la base de producción reventaba con
        `no such column`. Un chequeo de salud que se cae con una base vieja
        es lo contrario de lo que promete."""
        m = _metricas(_base(tmp_path, con_poll_attempts=False), _Settings(), dias=7)

        assert m["trabajos"]["en_seguimiento"] == 1
        assert "sin_noticias" not in m["trabajos"]

    def test_sin_base_lo_dice_en_vez_de_romperse(self, tmp_path):
        m = _metricas(tmp_path / "no-existe.db", _Settings(), dias=7)
        assert "error" in m

    def test_la_ventana_recorta(self, tmp_path):
        db = _base(tmp_path)
        viejo = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        with sqlite3.connect(db) as c:
            c.execute("INSERT INTO decisiones_router VALUES (?,?,?)", (9.9, "routed", viejo))

        assert _metricas(db, _Settings(), dias=7)["router"]["decisiones"] == 4
        assert _metricas(db, _Settings(), dias=365)["router"]["decisiones"] == 5


def _datos(**cambios) -> dict:
    base = {
        "ollama": {"ok": True, "detalle": "ok"},
        "cluster": {"ok": True, "detalle": "ok"},
        "metricas": {"router": {"p90": 30.0, "timeout": 180.0}},
    }
    base.update(cambios)
    return base


class TestDiagnostico:
    def test_todo_bien_no_reporta_nada(self):
        assert _diagnostico(_datos()) == []

    @pytest.mark.parametrize("dependencia", ["ollama", "cluster"])
    def test_una_dependencia_caida_se_reporta(self, dependencia):
        avisos = _diagnostico(_datos(**{dependencia: {"ok": False, "detalle": "no responde"}}))
        assert any(dependencia in a for a in avisos)

    def test_avisa_ANTES_de_que_el_p90_alcance_el_timeout(self):
        """El punto del umbral. Con el timeout en 120 s el p90 real llegó a
        118.1 (98%) y nadie se enteró hasta que se leyó la base a mano; para
        entonces ya se habían perdido pedidos."""
        avisos = _diagnostico(_datos(metricas={"router": {"p90": 118.1, "timeout": 120.0}}))
        assert any("p90" in a for a in avisos)

    def test_con_margen_no_molesta(self):
        assert _diagnostico(_datos(metricas={"router": {"p90": 118.1, "timeout": 300.0}})) == []

    def test_cuenta_los_fallos_del_modelo(self):
        avisos = _diagnostico(_datos(metricas={"fallos_del_modelo": {"timeout": 2, "unreachable": 1}}))
        assert any("3 vez/veces" in a for a in avisos)

    def test_los_trabajos_sin_noticias_se_reportan(self):
        avisos = _diagnostico(_datos(metricas={"trabajos": {"sin_noticias": 2}}))
        assert any("sin poder leer su estado" in a for a in avisos)

    def test_una_base_vieja_sin_esa_clave_no_rompe_el_diagnostico(self):
        assert _diagnostico(_datos(metricas={"trabajos": {"en_seguimiento": 1}})) == []
