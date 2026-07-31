"""Tests de los avisos de física.

Todas las combinaciones que se prueban acá generan un INCAR válido y una
corrida que VASP acepta sin quejarse. El punto es exactamente ese: son los
errores que no fallan. Las reglas salen del texto de §6.38 del manual, y cada
aviso cita la sección para que la afirmación sea verificable.
"""
from __future__ import annotations

import pytest

from becario.domain.models import CalcKind
from becario.domain.reglas_fisicas import advertencias


def _texto(*args, **kw) -> str:
    return "\n".join(advertencias(*args, **kw))


class TestTetraedrosAlRelajar:
    """"For relaxations *in metals* always use ISMEAR=1 or ISMEAR=2" (§6.38)."""

    @pytest.mark.parametrize("ismear", ["-5", "-4"])
    def test_avisa_al_relajar_con_tetraedros(self, ismear):
        t = _texto(CalcKind.RELAX, {"ISMEAR": ismear})
        assert "fuerzas salen mal" in t
        assert "§6.38" in t

    def test_no_avisa_en_un_estatico(self):
        """Para DOS y energías totales precisas los tetraedros son LO
        recomendado: el aviso es sobre relajar, no sobre el método."""
        assert advertencias(CalcKind.STATIC, {"ISMEAR": "-5"}) == []

    @pytest.mark.parametrize("ismear", ["1", "2"])
    def test_no_avisa_con_lo_que_el_manual_recomienda(self, ismear):
        assert advertencias(CalcKind.RELAX, {"ISMEAR": ismear}) == []

    def test_sin_tags_pedidos_no_hay_avisos(self):
        assert advertencias(CalcKind.RELAX) == []
        assert advertencias(CalcKind.RELAX, {}) == []


class TestSigmaIgnorado:
    def test_avisa_que_sigma_no_se_usa_con_tetraedros(self):
        t = _texto(CalcKind.STATIC, {"ISMEAR": "-5", "SIGMA": "0.05"})
        assert "no se usa" in t

    def test_sin_sigma_no_dice_nada(self):
        assert advertencias(CalcKind.STATIC, {"ISMEAR": "-5"}) == []


class TestMallaDeKpoints:
    """Los tetraedros necesitan una malla; con un solo punto k el manual
    manda ISMEAR=0."""

    def test_avisa_con_un_solo_punto_k(self):
        t = _texto(CalcKind.STATIC, {"ISMEAR": "-5"}, kpoints=(1, 1, 1))
        assert "un solo punto k" in t
        assert "ISMEAR=0" in t

    def test_no_avisa_con_una_malla_de_verdad(self):
        assert advertencias(CalcKind.STATIC, {"ISMEAR": "-5"}, kpoints=(6, 6, 6)) == []

    def test_sin_kpoints_no_adivina(self):
        """Si no se sabe la grilla, la regla se omite en vez de inventar."""
        assert advertencias(CalcKind.STATIC, {"ISMEAR": "-5"}, kpoints=None) == []


class TestGaussianoSinSigma:
    def test_avisa_que_queda_el_sigma_de_metales(self):
        t = _texto(CalcKind.STATIC, {"ISMEAR": "0"})
        assert "0.2 del" in t and "aislantes" in t

    def test_con_sigma_explicito_no_dice_nada(self):
        assert advertencias(CalcKind.STATIC, {"ISMEAR": "0", "SIGMA": "0.01"}) == []


class TestFormaDeLosAvisos:
    def test_todos_citan_la_seccion_del_manual(self):
        casos = [
            (CalcKind.RELAX, {"ISMEAR": "-5"}, None),
            (CalcKind.STATIC, {"ISMEAR": "-5", "SIGMA": "0.05"}, None),
            (CalcKind.STATIC, {"ISMEAR": "-5"}, (1, 1, 1)),
            (CalcKind.STATIC, {"ISMEAR": "0"}, None),
        ]
        for kind, tags, kp in casos:
            for aviso in advertencias(kind, tags, kp):
                assert "§" in aviso, aviso

    def test_no_bloquean_solo_avisan(self):
        """`advertencias` devuelve texto; nunca levanta. La decisión es de
        quien hace la física, igual que con el CONTCAR no convergido."""
        assert isinstance(advertencias(CalcKind.RELAX, {"ISMEAR": "-5"}), list)

    def test_el_valor_se_lee_aunque_venga_con_espacios(self):
        assert advertencias(CalcKind.RELAX, {"ISMEAR": " -5 "}) != []


class TestLleganALaConfirmacion:
    """El aviso solo sirve ANTES de confirmar: después ya se gastaron horas
    de cluster en una corrida que el manual desaconseja."""

    def test_el_texto_de_confirmacion_incluye_el_aviso(self, tmp_path):
        from types import SimpleNamespace

        from becario.domain.models import ClusterIdentity, VaspCalcRequest
        from becario.application.context import _Ctx
        from becario.application.handlers import calc as calc_mod
        from becario.infrastructure.vasp_inputs import VaspInputGenerator

        identity = ClusterIdentity(
            telegram_user_id=1, ssh_user="alice", ssh_key_path="/k", display_name="A",
        )

        class _Gateway:
            def make_directory(self, path):
                return SimpleNamespace(ok=True, message="ok")

            def upload_dir(self, local, remote):
                return SimpleNamespace(ok=True, message="ok")

            def home_dir(self):
                return "/home/alice"

            def file_exists(self, p):
                return True

            def concat_files(self, sources, dest):
                return SimpleNamespace(ok=True, message="ok")

        svc = SimpleNamespace(
            _calc_inputs=VaspInputGenerator(workdir=str(tmp_path)),
            _potcar_dir="/opt/potcar", _remote_base="runs", _calc_runs=None,
            _structure_provider=None, _mp_api_key="",
            _POTCAR_VARIANTS=("_sv", "_pv", ""),
        )
        ctx = _Ctx(chat_id=1, user_id=1, identity=identity, cluster=_Gateway())
        req = VaspCalcRequest(
            formula="W", calc_kind=CalcKind.RELAX, incar_tags={"ISMEAR": "-5"},
        )
        out = calc_mod._generate_and_upload(svc, ctx, req)
        # Que sea un _PreparedCalc y no un Reply: si el camino feliz se
        # rompiera, un getattr defensivo dejaría pasar el test en silencio.
        assert isinstance(out, calc_mod._PreparedCalc), getattr(out, "text", out)
        assert "fuerzas salen mal" in out.description
        assert "§6.38" in out.description
