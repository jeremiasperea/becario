"""Tests del vocabulario de tags del INCAR.

El test que importa es `test_todo_tag_emitido_esta_en_el_vocabulario`: ata lo
que el generador escribe a lo que el manual documenta. Un typo en una tabla
del generador no rompe VASP —lo ignora y corre con el default— así que sin
esto el error viaja hasta el resultado sin que nada se queje.
"""
from __future__ import annotations

import logging

import pytest

from becario.domain.vasp_tags import (
    TAGS_RESERVADOS,
    describir,
    es_tag_conocido,
    seccion_de,
    tags_desconocidos,
    total_tags,
)
from becario.domain.models import CalcKind, VaspCalcRequest
from becario.infrastructure.vasp_inputs import (
    _INCAR_COMMON,
    VaspInputGenerator,
)


class TestVocabulario:
    def test_tiene_los_tags_del_manual(self):
        assert total_tags() > 150

    @pytest.mark.parametrize(
        "tag", ["ENCUT", "ISMEAR", "SIGMA", "NSW", "IBRION", "EDIFFG", "ISIF",
                "PREC", "LREAL", "NPAR", "LMAXMIX", "AEXX", "ADDGRID"],
    )
    def test_conoce_los_tags_frecuentes(self, tag):
        assert es_tag_conocido(tag)

    @pytest.mark.parametrize("basura", ["ISMAER", "ENCUTT", "NOEXISTE", "ELASTIC", "TOTAL"])
    def test_rechaza_lo_que_no_es_un_tag(self, basura):
        """Incluye palabras inglesas que el manual pone en títulos y que un
        parser ingenuo confunde con tags."""
        assert not es_tag_conocido(basura)

    def test_es_insensible_a_mayusculas_y_espacios(self):
        assert es_tag_conocido(" encut ")

    def test_cita_la_seccion_del_manual(self):
        assert "6.38" in (seccion_de("ISMEAR") or "")
        assert "6.9" in (seccion_de("ENCUT") or "")

    def test_describe_los_que_estan_en_la_tabla_de_parametros(self):
        assert "cutoff" in describir("ENCUT").lower()

    def test_tags_desconocidos_conserva_orden_y_no_repite(self):
        assert tags_desconocidos(["ENCUT", "ISMAER", "ISMAER", "NOPE"]) == ["ISMAER", "NOPE"]


class TestTagsReservados:
    """NSW, IBRION e ISIF llevan reglas de física ya resueltas. Quedan fuera
    del alcance de cualquier override suelto para que no se las saltee."""

    @pytest.mark.parametrize("tag", ["NSW", "IBRION", "ISIF"])
    def test_los_de_regla_fisica_estan_reservados(self, tag):
        assert tag in TAGS_RESERVADOS

    def test_los_reservados_son_tags_reales(self):
        assert not tags_desconocidos(TAGS_RESERVADOS)


class TestGeneradorContraVocabulario:
    @pytest.fixture()
    def generator(self, tmp_path):
        return VaspInputGenerator(workdir=str(tmp_path))

    def test_todo_tag_emitido_esta_en_el_vocabulario(self, generator, tmp_path):
        """Ata el generador al manual: si alguien agrega un tag con un typo a
        `_INCAR_COMMON`, VASP lo ignoraría en silencio y esto lo caza."""
        vistos: set[str] = set()
        for i, kind in enumerate(CalcKind):
            gen = VaspInputGenerator(workdir=str(tmp_path / f"g{i}"))
            req = VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23,
                                  calc_kind=kind)
            result = gen.generate(req)
            from pathlib import Path

            for incar in Path(result.local_dir).rglob("INCAR"):
                for linea in incar.read_text().splitlines():
                    if "=" in linea:
                        vistos.add(linea.split("=")[0].strip())
        assert vistos, "no se leyó ningún INCAR"
        assert tags_desconocidos(vistos) == []

    def test_los_tags_de_calidad_fijos_son_reales(self):
        assert tags_desconocidos(_INCAR_COMMON) == []

    def test_un_tag_inventado_queda_registrado(self, generator, caplog, monkeypatch):
        """La guarda no rompe la corrida —el INCAR igual se escribe— pero deja
        rastro: preferimos un input generado y un warning a un fallo duro."""
        import becario.infrastructure.vasp_inputs as vi

        monkeypatch.setitem(vi._INCAR_COMMON, "ISMAER", 1)
        with caplog.at_level(logging.WARNING):
            generator.generate(VaspCalcRequest(formula="Zr", crystal="hcp", lattice_a=3.23))
        assert "ISMAER" in caplog.text
        assert "no documenta" in caplog.text


class TestCita:
    """La cita existe para que quien lee pueda verificar la afirmación, no
    para que el bot demuestre erudición."""

    def test_devuelve_la_seccion_lista_para_pegar(self):
        from becario.domain.vasp_tags import cita

        assert cita("NSW") == " (§6.20)"

    def test_usa_solo_la_seccion_principal(self):
        """IBRION figura en once secciones; once referencias al final de un
        aviso dejan de ser una cita."""
        from becario.domain.vasp_tags import cita, seccion_de

        assert seccion_de("IBRION").count("§") > 5
        assert cita("IBRION").count("§") == 1

    def test_vacia_si_no_conoce_el_tag(self):
        from becario.domain.vasp_tags import cita

        assert cita("NOEXISTE") == ""
