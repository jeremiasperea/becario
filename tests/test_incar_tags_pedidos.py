"""Tests de los tags del INCAR pedidos a mano.

Este camino deja que un mensaje de Telegram escriba en un INCAR, así que lo
que se prueba acá es sobre todo lo que NO debe pasar: que un tag mal escrito
llegue al archivo, que alguien se saltee las reglas del tipo de cálculo, o
que un valor con un salto de línea inyecte una segunda asignación.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from becario.domain.models import CalcKind, VaspCalcRequest
from becario.domain.vasp_tags import TAGS_RESERVADOS, sugerir, validar_tags_pedidos
from becario.infrastructure.vasp_inputs import VaspInputGenerator


@pytest.fixture()
def generator(tmp_path):
    return VaspInputGenerator(workdir=str(tmp_path))


_n = 0


def _incar(tmp_path, **kw):
    """INCAR de un pedido, cada uno en su propio workdir.

    Un generador por llamada a propósito: el nombre del directorio de corrida
    tiene resolución de UN SEGUNDO y `mkdir(exist_ok=False)`, así que dos
    pedidos del mismo material y tipo dentro del mismo segundo chocan. Lo
    arregla el PR #17, que no está en esta rama.
    """
    global _n
    _n += 1
    gen = VaspInputGenerator(workdir=str(Path(tmp_path) / f"run{_n}"))
    req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23, **kw)
    return (Path(gen.generate(req).local_dir) / "INCAR").read_text()


def _sin_system(incar: str) -> list[str]:
    """Las líneas del INCAR salvo SYSTEM, que lleva el sello de tiempo."""
    return [l for l in incar.splitlines() if not l.startswith("SYSTEM")]


class TestNoSeCuelaNadaAlIncar:
    """El valor se escribe tal cual en el archivo: ahí está el riesgo."""

    def test_un_salto_de_linea_no_puede_inyectar_otra_asignacion(self):
        """Sin esto, `{"LREAL": ".FALSE.\\nNSW = 999"}` mete un NSW por la
        ventana y se saltea la lista de reservados."""
        with pytest.raises(ValidationError, match="valor inválido"):
            VaspCalcRequest(formula="Zr", incar_tags={"LREAL": ".FALSE.\nNSW = 999"})

    def test_un_igual_en_el_valor_tampoco(self):
        with pytest.raises(ValidationError, match="valor inválido"):
            VaspCalcRequest(formula="Zr", incar_tags={"LREAL": ".FALSE. = 1"})

    @pytest.mark.parametrize("valor", [".TRUE.", "-5", "0.05", "Accurate", "1 1 1", "1E-6"])
    def test_los_valores_normales_de_vasp_pasan(self, valor):
        assert validar_tags_pedidos({"LREAL": valor}) == {"LREAL": valor}

    def test_el_valor_no_puede_ser_infinito(self):
        with pytest.raises(ValidationError, match="valor inválido"):
            VaspCalcRequest(formula="Zr", incar_tags={"LREAL": "x" * 200})


class TestSeRechazaLoQueNoCorresponde:
    @pytest.mark.parametrize("tag", sorted(TAGS_RESERVADOS))
    def test_los_que_decide_el_tipo_de_calculo(self, tag):
        """NSW, IBRION e ISIF llevan reglas ya resueltas; pedirlos a mano
        sería saltearlas por la puerta de atrás."""
        with pytest.raises(ValidationError, match="tipo de cálculo|campo"):
            VaspCalcRequest(formula="Zr", incar_tags={tag: "1"})

    @pytest.mark.parametrize("tag", ["ENCUT", "NBANDS", "ISPIN"])
    def test_los_que_ya_tienen_campo_propio(self, tag):
        with pytest.raises(ValidationError, match="campo"):
            VaspCalcRequest(formula="Zr", incar_tags={tag: "600"})

    def test_ispin_a_mano_no_puede_desmagnetizar_en_silencio(self):
        """Pedir un cálculo magnético y a la vez `ISPIN=1` a mano es una
        contradicción que cambia la FÍSICA de la corrida: el INCAR saldría
        sin polarización de espín aunque el usuario la pidió.

        Los tags a mano se aplican después de los de calidad, así que sin
        esta validación el tag ganaba en silencio. Es el mismo riesgo que
        motivó `descartar_numeros_inventados`: un valor que no falla, corre,
        y da otra física."""
        with pytest.raises(ValidationError, match="campo"):
            VaspCalcRequest(formula="Fe", ispin=2, incar_tags={"ISPIN": "1"})

    def test_ispin_a_mano_que_dice_lo_mismo_no_es_contradiccion(self):
        # Repetir un valor no es contradecirlo (PR #30): se ignora por
        # redundante en vez de rebotar el pedido entero.
        req = VaspCalcRequest(formula="Fe", ispin=2, incar_tags={"ISPIN": "2"})
        assert req.incar_tags == {}
        assert req.ispin == 2

    def test_un_tag_inexistente_no_se_ignora_se_rechaza(self):
        """VASP ignoraría el tag y correría con el default: el error tiene que
        aparecer acá o no aparece nunca."""
        with pytest.raises(ValidationError, match="no es un tag"):
            VaspCalcRequest(formula="Zr", incar_tags={"NOEXISTE": "1"})

    def test_un_typo_sugiere_el_tag_correcto(self):
        with pytest.raises(ValidationError, match="ISMEAR"):
            VaspCalcRequest(formula="Zr", incar_tags={"ISMAER": "0"})

    def test_la_sugerencia_no_inventa_cuando_no_hay_nada_parecido(self):
        assert sugerir("QQQQZZZZ") is None


class TestLlegaAlIncar:
    def test_un_tag_pedido_pisa_el_default_de_calidad(self, tmp_path):
        """Para eso se piden: para poder desviarse del template."""
        assert "ISMEAR = 1" in _incar(tmp_path)
        assert "ISMEAR = -5" in _incar(tmp_path, incar_tags={"ISMEAR": "-5"})

    def test_un_tag_que_el_template_no_trae_se_agrega(self, tmp_path):
        incar = _incar(tmp_path, incar_tags={"LORBIT": "11", "NEDOS": "3001"})
        assert "LORBIT = 11" in incar and "NEDOS = 3001" in incar

    def test_se_normaliza_a_mayusculas(self, tmp_path):
        assert "LORBIT = 11" in _incar(tmp_path, incar_tags={"lorbit": " 11 "})

    def test_tambien_en_cada_punto_del_barrido_de_encut(self, generator):
        req = VaspCalcRequest(
            formula="Zr", crystal="hcp", lattice_a=3.23,
            calc_kind=CalcKind.ENCUT_SCAN, encut_values=[300, 400],
            incar_tags={"LORBIT": "11"},
        )
        run = Path(generator.generate(req).local_dir)
        for encut in (300, 400):
            assert "LORBIT = 11" in (run / f"encut_{encut}" / "INCAR").read_text()

    def test_no_dispara_la_guarda_de_vocabulario(self, caplog, tmp_path):
        """Un tag válido pedido a mano ya pasó por el vocabulario: la guarda
        del generador no tiene nada que decir."""
        import logging

        with caplog.at_level(logging.WARNING):
            _incar(tmp_path, incar_tags={"LORBIT": "11"})
        assert "no documenta" not in caplog.text

    def test_sin_tags_el_incar_no_cambia(self, tmp_path):
        """El cambio es aditivo: sin tags pedidos, el INCAR es el de siempre."""
        base = _incar(tmp_path)
        vacio = _incar(tmp_path, incar_tags={})
        assert _sin_system(base) == _sin_system(vacio)


class TestRouterYHandler:
    """Un campo en el schema (ADR-0006), no uno por tag."""

    def test_el_schema_tiene_el_campo_generico(self):
        from becario.infrastructure.ollama_router import RouterParams

        assert "tags_incar" in RouterParams.model_json_schema()["properties"]

    def test_el_handler_reenvia_los_tags_al_pedido(self):
        from types import SimpleNamespace

        from becario.application.handlers.calc import _build_calc_request

        svc = SimpleNamespace(_calc_inputs=object(), _potcar_dir="/opt/potcar")
        req = _build_calc_request(svc, {
            "formula": "Zr", "tipo_calculo": "estatico",
            "tags_incar": {"ISMEAR": "-5", "LORBIT": "11"},
        })
        assert isinstance(req, VaspCalcRequest)
        assert req.incar_tags == {"ISMEAR": "-5", "LORBIT": "11"}

    def test_un_tag_invalido_del_router_devuelve_aviso_no_excepcion(self):
        from types import SimpleNamespace

        from becario.application.context import Reply
        from becario.application.handlers.calc import _build_calc_request

        svc = SimpleNamespace(_calc_inputs=object(), _potcar_dir="/opt/potcar")
        out = _build_calc_request(svc, {
            "formula": "Zr", "tipo_calculo": "estatico",
            "tags_incar": {"ISMAER": "0"},
        })
        assert isinstance(out, Reply)
        assert "ISMEAR" in out.text          # la sugerencia llega al usuario

    def test_sin_tags_el_pedido_queda_vacio(self):
        from types import SimpleNamespace

        from becario.application.handlers.calc import _build_calc_request

        svc = SimpleNamespace(_calc_inputs=object(), _potcar_dir="/opt/potcar")
        req = _build_calc_request(svc, {"formula": "Zr"})
        assert req.incar_tags == {}
