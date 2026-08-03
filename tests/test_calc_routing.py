"""Tests del ruteo de fuente de estructura en la capa de aplicación (PR2).

Cubre R1 (elemento -> ASE, sin key), R2 (compuesto -> MP por fórmula),
R4 (mp-id fuerza MP), R5 (falta key -> fail-closed) y R6/R7 (errores de MP).
Se prueba `_resolve_structure` directo con un `svc` stub — sin cluster.
"""
from __future__ import annotations

from types import SimpleNamespace

from becario.application.context import Reply
from becario.application.handlers.calc import (
    _build_calc_request,
    _mp_note,
    _resolve_structure,
)
from becario.domain.models import (
    StructureAlternative,
    StructureResolution,
    StructureResolutionError,
    StructureResolutionReason,
    StructureSource,
    VaspCalcRequest,
)

from .fakes import FakeStructureProvider


def _resolution(energy_above_hull):
    return StructureResolution(
        atoms=None,
        mp_id="mp-19770",
        formula="Fe2O3",
        spacegroup="R-3c",
        energy_above_hull=energy_above_hull,
    )


class TestMpNote:
    """La nota humana no debe afirmar estabilidad que no conoce: pedir un
    polimorfo por su mp-id deja el hull en None (no hay summary), y la nota
    NO puede decir "la más estable" ni inventar E_hull=0.000."""

    def test_note_omits_stability_when_hull_unknown(self):
        note = _mp_note(_resolution(None))
        assert "más estable" not in note
        assert "E_hull" not in note
        assert "mp-19770" in note

    def test_note_reports_hull_when_known(self):
        note = _mp_note(_resolution(0.008))
        assert "más estable" in note
        assert "E_hull=0.008" in note


def _zro2_resolution(chosen="Monoclinic", others=("Tetragonal", "Cubic")):
    """Resolución de un compuesto con polimorfos, como la devuelve MP: gana
    el del hull y los demás viajan en `alternatives`."""
    return StructureResolution(
        atoms=None,
        mp_id="mp-2858",
        formula="ZrO2",
        spacegroup="P2_1/c",
        crystal_system=chosen,
        energy_above_hull=0.0,
        alternatives=tuple(
            StructureAlternative(
                mp_id=f"mp-{i}", formula="ZrO2",
                energy_above_hull=0.04 * (i + 1), crystal_system=s,
            )
            for i, s in enumerate(others)
        ),
        # `phases` las trae todas; `alternatives` viene recortada por
        # estabilidad. La pregunta se arma con las primeras.
        phases=(chosen,) + tuple(others),
    )


class TestPhaseIsAsked:
    """Un compuesto con varias fases: se pregunta en vez de devolver la del
    hull sin avisar. Para ZrO2 la del hull es la monoclínica —correcta a
    temperatura ambiente— pero quien busca la tetragonal de un recubrimiento
    recibiría otra estructura sin enterarse."""

    def test_asks_when_there_are_other_phases(self):
        fake = FakeStructureProvider(resolution=_zro2_resolution())
        out = _resolve_structure(_svc(fake), _ctx(), VaspCalcRequest(formula="ZrO2"))
        assert isinstance(out, Reply)
        assert out.awaiting_params
        # Las fases se ofrecen por su nombre, que es como se las pide.
        assert "monoclínica" in out.text
        assert "tetragonal" in out.text
        assert "cúbica" in out.text

    def test_requested_phase_travels_to_the_query(self):
        fake = FakeStructureProvider(resolution=_zro2_resolution("Tetragonal", ()))
        out = _resolve_structure(
            _svc(fake), _ctx(), VaspCalcRequest(formula="ZrO2", crystal="tetragonal")
        )
        assert not isinstance(out, Reply)
        assert fake.calls[0].crystal_system == "Tetragonal"

    def test_single_phase_does_not_ask(self):
        fake = FakeStructureProvider(resolution=_zro2_resolution("Monoclinic", ()))
        out = _resolve_structure(_svc(fake), _ctx(), VaspCalcRequest(formula="ZrO2"))
        assert not isinstance(out, Reply)

    def test_mp_id_is_an_explicit_choice_and_never_asks(self):
        # Pedir un polimorfo por su id ya ES elegir la fase.
        fake = FakeStructureProvider(resolution=_zro2_resolution())
        out = _resolve_structure(
            _svc(fake), _ctx(), VaspCalcRequest(formula="ZrO2", mp_id="mp-1565")
        )
        assert not isinstance(out, Reply)

    def test_offers_phases_that_the_alternatives_would_hide(self):
        """Las fases salen de `phases`, no de `alternatives`.

        Caso real de MP: ZrO2 tiene 20 estructuras y la cúbica es la 16ª por
        energía, así que no entra en las 5 alternativas. Si la pregunta se
        armara con ellas, escondería justo la fase tipo fluorita."""
        res = StructureResolution(
            atoms=None, mp_id="mp-2858", formula="ZrO2", spacegroup="P2_1/c",
            crystal_system="Monoclinic", energy_above_hull=0.0,
            alternatives=(
                StructureAlternative("mp-776404", "ZrO2", 0.0095, "Orthorhombic"),
            ),
            phases=("Cubic", "Monoclinic", "Orthorhombic", "Tetragonal"),
        )
        out = _resolve_structure(
            _svc(FakeStructureProvider(resolution=res)), _ctx(),
            VaspCalcRequest(formula="ZrO2"),
        )
        assert isinstance(out, Reply)
        assert "cúbica" in out.text      # no está entre las alternativas
        assert "tetragonal" in out.text  # tampoco

    def test_prototype_is_not_a_phase_filter(self):
        # 'fluorita' es un prototipo de ASE, no un sistema cristalino: no
        # debe viajar como filtro de fase a MP.
        fake = FakeStructureProvider(resolution=_zro2_resolution("Cubic", ()))
        _resolve_structure(
            _svc(fake), _ctx(),
            VaspCalcRequest(formula="ZrO2", crystal="fluorita", lattice_a=5.07),
        )
        assert fake.calls[0].crystal_system is None


def _svc(provider=None, key="secret", calc_runs=None):
    return SimpleNamespace(
        _structure_provider=provider, _mp_api_key=key, _calc_runs=calc_runs
    )


def _ctx(cluster=None):
    """Los caminos ASE y MP no tocan el contexto; el de `relajado` sí,
    porque lee del cluster con la cuenta de quien pide."""
    return SimpleNamespace(user_id=111, cluster=cluster)


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
        atoms, note = _resolve_structure(_svc(fake), _ctx(), VaspCalcRequest(formula="Zr"))
        assert atoms is None  # ASE: el generador la arma
        assert note == ""
        assert fake.calls == []  # MP nunca se consultó

    def test_single_element_needs_no_key(self):
        # sin provider ni key, un elemento simple sigue funcionando (R1/R5)
        atoms, note = _resolve_structure(
            _svc(provider=None, key=""), _ctx(), VaspCalcRequest(formula="W")
        )
        assert atoms is None

    def test_source_ase_forces_ase_even_for_compound(self):
        fake = FakeStructureProvider()
        atoms, _ = _resolve_structure(
            _svc(fake), _ctx(), VaspCalcRequest(formula="Fe2O3", source="ase")
        )
        assert atoms is None
        assert fake.calls == []


class TestMpPath:
    def test_compound_queries_mp_by_formula(self):
        fake = FakeStructureProvider()
        atoms, note = _resolve_structure(_svc(fake), _ctx(), VaspCalcRequest(formula="Fe2O3"))
        assert atoms is not None
        assert fake.calls[0].formula == "Fe2O3"
        assert "Materials Project" in note

    def test_mp_id_forces_mp_by_id(self):
        fake = FakeStructureProvider()
        _resolve_structure(
            _svc(fake), _ctx(), VaspCalcRequest(formula="Fe2O3", mp_id="mp-19770")
        )
        assert fake.calls[0].mp_id == "mp-19770"

    def test_source_mp_on_single_element(self):
        fake = FakeStructureProvider()
        atoms, _ = _resolve_structure(
            _svc(fake), _ctx(), VaspCalcRequest(formula="Fe", source="mp")
        )
        assert atoms is not None
        assert fake.calls[0].formula == "Fe"


class TestFailClosed:
    def test_missing_key_is_reply(self):
        out = _resolve_structure(
            _svc(FakeStructureProvider(), key=""), _ctx(), VaspCalcRequest(formula="Fe2O3")
        )
        assert isinstance(out, Reply)
        assert "BECARIO_MP_API_KEY" in out.text

    def test_no_provider_is_reply(self):
        out = _resolve_structure(
            _svc(provider=None, key="k"), _ctx(), VaspCalcRequest(formula="Fe2O3")
        )
        assert isinstance(out, Reply)

    def test_network_error_maps_to_reply(self):
        fake = FakeStructureProvider(
            error=StructureResolutionError(StructureResolutionReason.NETWORK)
        )
        out = _resolve_structure(_svc(fake), _ctx(), VaspCalcRequest(formula="Fe2O3"))
        assert isinstance(out, Reply)
        assert "conectarme" in out.text

    def test_no_match_maps_to_reply(self):
        fake = FakeStructureProvider(
            error=StructureResolutionError(StructureResolutionReason.NO_MATCH)
        )
        out = _resolve_structure(_svc(fake), _ctx(), VaspCalcRequest(formula="Fe2O3"))
        assert isinstance(out, Reply)
        assert "encontré" in out.text


class TestRelaxedPath:
    """`fuente_estructura=relajado`: partir del CONTCAR propio anterior.

    El preflight vive en `relaxed_source` (ver su test); acá se verifica el
    empalme — que esta rama se elija, que corte ANTES de generar nada, y que
    el aviso de no-convergencia viaje hasta el texto que el usuario confirma.
    """

    def _req(self, **kw):
        return VaspCalcRequest(formula="Zr", source="relajado", **kw)

    def test_relaxed_source_never_touches_materials_project(self):
        from tests.test_relaxed_source import FakeCalcRuns, FakeCluster

        fake = FakeStructureProvider()
        atoms, note = _resolve_structure(
            _svc(fake, calc_runs=FakeCalcRuns()),
            _ctx(cluster=FakeCluster()),
            self._req(),
        )
        assert atoms is not None
        assert fake.calls == [], "es un resultado propio, no una fuente externa"
        assert "CONTCAR" in note

    def test_preflight_failure_is_a_reply_not_an_exception(self):
        """Fail-closed: devuelve un aviso, y `_generate_and_upload` corta ahí
        sin haber creado ningún directorio de corrida."""
        from tests.test_relaxed_source import FakeCalcRuns, FakeCluster

        out = _resolve_structure(
            _svc(calc_runs=FakeCalcRuns(rows=[])),
            _ctx(cluster=FakeCluster()),
            self._req(),
        )
        assert isinstance(out, Reply)
        assert "No encontré ninguna relajación" in out.text

    def test_running_job_blocks_with_a_clear_message(self):
        from tests.test_relaxed_source import FakeCalcRuns, FakeCluster

        out = _resolve_structure(
            _svc(calc_runs=FakeCalcRuns()),
            _ctx(cluster=FakeCluster(state="RUNNING")),
            self._req(),
        )
        assert isinstance(out, Reply)
        assert "todavía está corriendo" in out.text

    def test_non_converged_warning_reaches_the_note(self):
        """No bloquea, pero el aviso tiene que llegar al texto que se confirma:
        es ahí donde el usuario decide si igual quiere partir de esa celda."""
        from tests.test_relaxed_source import RUN_DIR, FakeCalcRuns, FakeCluster, _oszicar

        cluster = FakeCluster()
        cluster.files[f"{RUN_DIR}/OSZICAR"] = _oszicar(180)
        atoms, note = _resolve_structure(
            _svc(calc_runs=FakeCalcRuns()), _ctx(cluster=cluster), self._req()
        )
        assert atoms is not None
        assert "OJO" in note and "NO está relajada" in note


class TestMpNoteDoesNotOverclaimStability:
    """Con una fase pedida, lo elegido es lo más estable DE ESA FASE.

    Decir "la más estable" a secas se contradice con su propio número: si lo
    fuera, E_hull sería 0. Y es la última pantalla antes de confirmar un
    cálculo de días — el lugar donde una afirmación falsa sobre la física
    cuesta más caro.
    """

    def test_with_a_requested_phase_it_scopes_the_claim(self):
        note = _mp_note(_zro2_resolution("Tetragonal", ()), "Tetragonal")
        assert "la más estable de la fase tetragonal" in note

    def test_a_phase_above_the_hull_is_flagged(self):
        res = StructureResolution(
            atoms=None, mp_id="mp-2574", formula="ZrO2", spacegroup="P4_2/nmc",
            crystal_system="Tetragonal", energy_above_hull=0.027,
        )
        note = _mp_note(res, "Tetragonal")
        assert "no es el estado fundamental" in note
        assert "0.027" in note

    def test_the_ground_state_of_the_phase_is_not_flagged(self):
        res = StructureResolution(
            atoms=None, mp_id="mp-2858", formula="ZrO2", spacegroup="P2_1/c",
            crystal_system="Monoclinic", energy_above_hull=0.0,
        )
        assert "no es el estado fundamental" not in _mp_note(res, "Monoclinic")

    def test_without_a_phase_the_old_wording_stays(self):
        res = StructureResolution(
            atoms=None, mp_id="mp-19770", formula="Fe2O3", spacegroup="C2/m",
            energy_above_hull=0.0,
        )
        note = _mp_note(res)
        assert "la más estable (E_hull=0.000 eV/át.)" in note
