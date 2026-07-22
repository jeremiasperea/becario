"""Tests del puerto StructureProvider, sus DTOs de dominio y el doble de prueba.

PR1 de materials-project-provider: cubre R2/R3/R4 (formas de identificar un
material), R5 (config de la API key) y R6/R7 (excepción clasificada). El
adaptador real contra Materials Project se testea en slices siguientes.
"""
from __future__ import annotations

import pytest

from becario.config import Settings
from becario.domain.models import (
    StructureAlternative,
    StructureQuery,
    StructureRequest,
    StructureResolution,
    StructureResolutionError,
    StructureResolutionReason,
    StructureSource,
    VaspCalcRequest,
)
from becario.domain.ports import StructureProvider

from .fakes import FakeStructureProvider


class TestStructureQuery:
    def test_accepts_chemsys(self):
        q = StructureQuery(elements=("Fe", "O"), qualifier="O")
        assert q.elements == ("Fe", "O")
        assert q.qualifier == "O"

    def test_accepts_explicit_formula(self):
        assert StructureQuery(formula="Fe2O3").formula == "Fe2O3"

    def test_accepts_mp_id(self):
        assert StructureQuery(mp_id="mp-19770").mp_id == "mp-19770"

    def test_rejects_bad_mp_id(self):
        with pytest.raises(ValueError):
            StructureQuery(mp_id="19770")

    def test_rejects_evil_formula(self):
        with pytest.raises(ValueError):
            StructureQuery(formula="oxido; rm -rf /")

    def test_requires_at_least_one_identifier(self):
        with pytest.raises(ValueError):
            StructureQuery()


class TestStructureResolutionError:
    def test_carries_classified_reason(self):
        err = StructureResolutionError(StructureResolutionReason.NO_MATCH)
        assert err.reason is StructureResolutionReason.NO_MATCH

    def test_reason_enum_is_stable(self):
        assert {r.value for r in StructureResolutionReason} == {
            "network", "api", "no_match",
        }


class TestStructureSource:
    def test_values_and_default(self):
        assert StructureSource.AUTO.value == "auto"
        assert {s.value for s in StructureSource} == {"auto", "ase", "mp"}


class TestFakeProvider:
    def test_satisfies_protocol(self):
        provider: StructureProvider = FakeStructureProvider()
        res = provider.resolve(StructureQuery(formula="NaCl"))
        assert isinstance(res, StructureResolution)
        assert res.atoms.get_chemical_formula() == "ClNa"
        assert res.mp_id.startswith("mp-")

    def test_records_the_query(self):
        fake = FakeStructureProvider()
        q = StructureQuery(elements=("Na", "Cl"))
        fake.resolve(q)
        assert fake.calls == [q]

    def test_raises_configured_error(self):
        fake = FakeStructureProvider(
            error=StructureResolutionError(StructureResolutionReason.NETWORK)
        )
        with pytest.raises(StructureResolutionError):
            fake.resolve(StructureQuery(formula="NaCl"))

    def test_returns_configured_resolution_with_alternatives(self):
        from .fakes import demo_atoms

        res = StructureResolution(
            atoms=demo_atoms(),
            mp_id="mp-19770",
            formula="Fe2O3",
            spacegroup="R-3c",
            energy_above_hull=0.0,
            alternatives=(StructureAlternative("mp-1234", "Fe3O4", 0.02),),
        )
        out = FakeStructureProvider(resolution=res).resolve(
            StructureQuery(mp_id="mp-19770")
        )
        assert out.formula == "Fe2O3"
        assert out.alternatives[0].mp_id == "mp-1234"


class TestRequestMpFields:
    def test_vasp_request_defaults_are_inert(self):
        req = VaspCalcRequest(formula="Zr")
        assert req.mp_id is None
        assert req.source is StructureSource.AUTO

    def test_vasp_request_accepts_mp_id_and_source(self):
        req = VaspCalcRequest(formula="Fe2O3", mp_id="mp-19770", source="mp")
        assert req.mp_id == "mp-19770"
        assert req.source is StructureSource.MP

    def test_vasp_request_rejects_bad_mp_id(self):
        with pytest.raises(ValueError):
            VaspCalcRequest(formula="Fe2O3", mp_id="19770")

    def test_vasp_request_rejects_bad_source(self):
        with pytest.raises(ValueError):
            VaspCalcRequest(formula="Fe2O3", source="materials-project")

    def test_structure_request_accepts_mp_fields(self):
        req = StructureRequest(formula="Fe2O3", mp_id="mp-19770", source="mp")
        assert req.mp_id == "mp-19770"
        assert req.source is StructureSource.MP

    def test_structure_request_rejects_bad_mp_id(self):
        with pytest.raises(ValueError):
            StructureRequest(formula="Fe2O3", mp_id="xx-1")


class TestConfigMpApiKey:
    def test_field_defaults_empty(self):
        assert Settings.__dataclass_fields__["mp_api_key"].default == ""

    def test_from_env_reads_key(self, monkeypatch):
        # el entorno gana sobre el .env (load_local_env no pisa os.environ)
        monkeypatch.setenv("BECARIO_BOT_TOKEN", "token")
        monkeypatch.setenv("BECARIO_SSH_HOST", "host")
        monkeypatch.setenv("BECARIO_MP_API_KEY", "secret123")
        assert Settings.from_env().mp_api_key == "secret123"

    def test_from_env_defaults_empty_when_absent(self, monkeypatch):
        monkeypatch.setenv("BECARIO_BOT_TOKEN", "token")
        monkeypatch.setenv("BECARIO_SSH_HOST", "host")
        monkeypatch.delenv("BECARIO_MP_API_KEY", raising=False)
        assert Settings.from_env().mp_api_key == ""
