"""Tests del tipo de cálculo DOS.

Una DOS es un cálculo de un solo punto con dos diferencias que importan: se
muestrea la zona de Brillouin mucho más fino, y se usan tetraedros. El manual
lo dice explícito en §6.38 — "for the calculations of the DOS (...) use the
tetrahedron method" — y en §6.37 documenta NEDOS, la finura de la grilla de
energía, cuyo default (301) es demasiado grueso para mirar un pico.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from becario.domain.models import CalcKind, VaspCalcRequest
from becario.infrastructure.vasp_inputs import VaspInputGenerator


def _gen(tmp_path, n=""):
    return VaspInputGenerator(workdir=str(tmp_path / f"g{n}"))


def _incar(tmp_path, n="", **kw):
    req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23, **kw)
    return (Path(_gen(tmp_path, n).generate(req).local_dir) / "INCAR").read_text()


class TestIncarDeDos:
    def test_usa_tetraedros_como_manda_el_manual(self, tmp_path):
        assert "ISMEAR = -5" in _incar(tmp_path, calc_kind=CalcKind.DOS)

    def test_afina_la_grilla_de_energia(self, tmp_path):
        """El default de VASP es 301: demasiado grueso para resolver un pico."""
        assert "NEDOS = 3001" in _incar(tmp_path, calc_kind=CalcKind.DOS)

    def test_pide_la_proyeccion_por_sitio_y_orbital(self, tmp_path):
        """Sin LORBIT sale la DOS total; con 11, la proyectada, que es lo que
        se quiere al preguntar por la DOS de un elemento en una interfaz."""
        assert "LORBIT = 11" in _incar(tmp_path, calc_kind=CalcKind.DOS)

    def test_es_de_un_solo_punto(self, tmp_path):
        incar = _incar(tmp_path, calc_kind=CalcKind.DOS)
        assert "NSW = 0" in incar
        assert "IBRION" not in incar and "EDIFFG" not in incar

    def test_no_toca_los_otros_tipos(self, tmp_path):
        """El estático sigue con el smearing de metales del template."""
        incar = _incar(tmp_path, n="s", calc_kind=CalcKind.STATIC)
        assert "ISMEAR = 1" in incar
        assert "NEDOS" not in incar and "LORBIT" not in incar


class TestMallaDeKpoints:
    def test_la_dos_se_muestrea_mas_fino_que_un_estatico(self, tmp_path):
        """La relajación necesita fuerzas, que convergen rápido con la malla;
        la DOS necesita resolver bandas, y con malla pobre salen picos que no
        existen."""
        req = dict(formula="Zr", crystal="hcp", lattice_a=3.23)
        dos = _gen(tmp_path, "d").generate(
            VaspCalcRequest(**req, calc_kind=CalcKind.DOS)
        )
        est = _gen(tmp_path, "e").generate(
            VaspCalcRequest(**req, calc_kind=CalcKind.STATIC)
        )
        assert all(d >= e for d, e in zip(dos.kpoints, est.kpoints))
        assert dos.kpoints != est.kpoints

    def test_una_grilla_pedida_a_mano_gana(self, tmp_path):
        r = _gen(tmp_path).generate(VaspCalcRequest(
            formula="Zr", crystal="hcp", lattice_a=3.23,
            calc_kind=CalcKind.DOS, kpoints=(4, 4, 4),
        ))
        assert r.kpoints == (4, 4, 4)


class TestElUsuarioSiemprePuedeDesviarse:
    def test_un_tag_pedido_pisa_el_default_del_tipo(self, tmp_path):
        """Los tags del tipo van ANTES que los pedidos a mano: si alguien
        pide ISMEAR=0 para una DOS de un aislante chico, gana el suyo."""
        incar = _incar(
            tmp_path, calc_kind=CalcKind.DOS, incar_tags={"ISMEAR": "0"},
        )
        assert "ISMEAR = 0" in incar
        assert "ISMEAR = -5" not in incar

    def test_y_puede_afinar_nedos(self, tmp_path):
        incar = _incar(
            tmp_path, calc_kind=CalcKind.DOS, incar_tags={"NEDOS": "6001"},
        )
        assert "NEDOS = 6001" in incar


class TestRuteo:
    def test_el_router_conoce_el_tipo(self):
        from becario.infrastructure.ollama_router import RouterParams

        desc = RouterParams.model_json_schema()["properties"]["tipo_calculo"]
        assert "dos" in desc["description"]

    def test_el_handler_lo_arma(self):
        from types import SimpleNamespace

        from becario.application.handlers.calc import _build_calc_request

        svc = SimpleNamespace(_calc_inputs=object(), _potcar_dir="/opt/potcar")
        req = _build_calc_request(svc, {"formula": "Zr", "tipo_calculo": "dos"})
        assert isinstance(req, VaspCalcRequest)
        assert req.calc_kind is CalcKind.DOS


class TestSinParametrosQueNoHacenNada:
    def test_la_dos_no_emite_sigma(self, tmp_path):
        """Con tetraedros el ancho de smearing no se usa (§6.38). Dejarlo
        escrito sugiere que hace algo — y el propio bot advierte de esa
        combinación cuando la pide un usuario."""
        assert "SIGMA" not in _incar(tmp_path, calc_kind=CalcKind.DOS)

    def test_los_demas_tipos_lo_conservan(self, tmp_path):
        assert "SIGMA = 0.2" in _incar(tmp_path, n="s", calc_kind=CalcKind.STATIC)

    def test_pero_se_puede_pedir_a_mano(self, tmp_path):
        """Si alguien pasa ISMEAR=0 y SIGMA, los dos tienen que salir."""
        incar = _incar(
            tmp_path, calc_kind=CalcKind.DOS,
            incar_tags={"ISMEAR": "0", "SIGMA": "0.02"},
        )
        assert "SIGMA = 0.02" in incar and "ISMEAR = 0" in incar
