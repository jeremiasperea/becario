"""Tests del ruteo de fuente de estructura en la capa de aplicación (PR2).

Cubre R1 (elemento -> ASE, sin key), R2 (compuesto -> MP por fórmula),
R4 (mp-id fuerza MP), R5 (falta key -> fail-closed) y R6/R7 (errores de MP).
Se prueba `_resolve_structure` directo con un `svc` stub — sin cluster.
"""
from __future__ import annotations

from types import SimpleNamespace

from becario.application.context import Reply
from becario.application.handlers.calc import _build_calc_request, _resolve_structure
from becario.domain.models import (
    StructureResolutionError,
    StructureResolutionReason,
    StructureSource,
    VaspCalcRequest,
)

from .fakes import FakeStructureProvider


def _svc(provider=None, key="secret"):
    return SimpleNamespace(_structure_provider=provider, _mp_api_key=key)


def _build_svc():
    """Stub mínimo para `_build_calc_request`: solo mira `_calc_inputs`
    (que exista) y `_potcar_dir` (ruta absoluta). Sin I/O al cluster."""
    return SimpleNamespace(_calc_inputs=object(), _potcar_dir="/opt/potcar")


class TestBuildForwardsStructureSource:
    """El router extrae `mp_id`/`fuente_estructura`; `_build_calc_request`
    debe reenviarlos al `VaspCalcRequest` para que el ruteo a MP los vea."""

    def test_forwards_mp_id(self):
        req = _build_calc_request(_build_svc(), {"formula": "Fe2O3", "mp_id": "mp-19770"})
        assert isinstance(req, VaspCalcRequest)
        assert req.mp_id == "mp-19770"

    def test_forwards_source_mp(self):
        req = _build_calc_request(_build_svc(), {"formula": "Fe", "fuente_estructura": "mp"})
        assert req.source is StructureSource.MP

    def test_forwards_source_ase(self):
        req = _build_calc_request(_build_svc(), {"formula": "Fe2O3", "fuente_estructura": "ase"})
        assert req.source is StructureSource.ASE

    def test_defaults_to_auto_without_hints(self):
        req = _build_calc_request(_build_svc(), {"formula": "W"})
        assert req.mp_id is None
        assert req.source is StructureSource.AUTO

    def test_unknown_source_falls_back_to_auto(self):
        # Un typo del LLM en la fuente no debe romper el cálculo.
        req = _build_calc_request(_build_svc(), {"formula": "W", "fuente_estructura": "xyz"})
        assert req.source is StructureSource.AUTO


class TestAsePath:
    def test_single_element_uses_ase(self):
        fake = FakeStructureProvider()
        atoms, note = _resolve_structure(_svc(fake), VaspCalcRequest(formula="Zr"))
        assert atoms is None  # ASE: el generador la arma
        assert note == ""
        assert fake.calls == []  # MP nunca se consultó

    def test_single_element_needs_no_key(self):
        # sin provider ni key, un elemento simple sigue funcionando (R1/R5)
        atoms, note = _resolve_structure(
            _svc(provider=None, key=""), VaspCalcRequest(formula="W")
        )
        assert atoms is None

    def test_source_ase_forces_ase_even_for_compound(self):
        fake = FakeStructureProvider()
        atoms, _ = _resolve_structure(
            _svc(fake), VaspCalcRequest(formula="Fe2O3", source="ase")
        )
        assert atoms is None
        assert fake.calls == []


class TestMpPath:
    def test_compound_queries_mp_by_formula(self):
        fake = FakeStructureProvider()
        atoms, note = _resolve_structure(_svc(fake), VaspCalcRequest(formula="Fe2O3"))
        assert atoms is not None
        assert fake.calls[0].formula == "Fe2O3"
        assert "Materials Project" in note

    def test_mp_id_forces_mp_by_id(self):
        fake = FakeStructureProvider()
        _resolve_structure(
            _svc(fake), VaspCalcRequest(formula="Fe2O3", mp_id="mp-19770")
        )
        assert fake.calls[0].mp_id == "mp-19770"

    def test_source_mp_on_single_element(self):
        fake = FakeStructureProvider()
        atoms, _ = _resolve_structure(
            _svc(fake), VaspCalcRequest(formula="Fe", source="mp")
        )
        assert atoms is not None
        assert fake.calls[0].formula == "Fe"


class TestFailClosed:
    def test_missing_key_is_reply(self):
        out = _resolve_structure(
            _svc(FakeStructureProvider(), key=""), VaspCalcRequest(formula="Fe2O3")
        )
        assert isinstance(out, Reply)
        assert "BECARIO_MP_API_KEY" in out.text

    def test_no_provider_is_reply(self):
        out = _resolve_structure(
            _svc(provider=None, key="k"), VaspCalcRequest(formula="Fe2O3")
        )
        assert isinstance(out, Reply)

    def test_network_error_maps_to_reply(self):
        fake = FakeStructureProvider(
            error=StructureResolutionError(StructureResolutionReason.NETWORK)
        )
        out = _resolve_structure(_svc(fake), VaspCalcRequest(formula="Fe2O3"))
        assert isinstance(out, Reply)
        assert "conectarme" in out.text

    def test_no_match_maps_to_reply(self):
        fake = FakeStructureProvider(
            error=StructureResolutionError(StructureResolutionReason.NO_MATCH)
        )
        out = _resolve_structure(_svc(fake), VaspCalcRequest(formula="Fe2O3"))
        assert isinstance(out, Reply)
        assert "encontré" in out.text
