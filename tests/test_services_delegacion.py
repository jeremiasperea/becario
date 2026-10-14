"""Contrato de la delegación que dejó el refactor de `services.py`.

`BecarioService` se partió en módulos (`textos`, `pending_edits`,
`plan_describer`, `plan_coordinator`) sin cambiar su superficie: los
nombres viejos se re-exportan, los helpers puros quedan como alias en la
clase y cada método movido deja un delegador fino. El resto de la suite
prueba el comportamiento de punta a punta; estos tests fijan las costuras
que ese comportamiento da por sentadas y que un refactor futuro podría
romper sin que nada más se entere:

- (a) los re-exports y alias son los MISMOS objetos que en su módulo nuevo;
- (b) cada delegador llama a `plan_coordinator.x(svc, ...)` con los mismos
  argumentos y devuelve lo que devuelve la función;
- (c) adentro de `plan_coordinator` las llamadas pasan por `svc.`: por eso
  parchear un método en la instancia sigue cambiando el flujo, y por eso
  el ciclo edición → `_dispatch_plan` sigue pasando por la clase.
"""
import inspect

import pytest

from becario.application import (
    pending_edits,
    plan_coordinator,
    plan_describer,
    services,
    textos,
)
from becario.application.context import Reply
from becario.application.handlers import remote_files
from becario.application.services import BecarioService
from becario.domain.models import (
    Intent,
    PendingEdit,
    Plan,
    RouterFailureReason,
    RouterUnavailableError,
)
from becario.infrastructure.storage import InMemoryConfirmationStore

from .test_service import (
    ALICE,
    FakeCalcInputGenerator,
    FakeCalcRuns,
    FakeClusterGatewayFactory,
    FakeHistory,
    FakeJobTracker,
    FakeRouter,
    FakeStructureBuilder,
    FakeUserRegistry,
    RoutedRequest,
)


@pytest.fixture()
def servicio():
    router = FakeRouter()
    service = BecarioService(
        router=router,
        registry=FakeUserRegistry([ALICE]),
        cluster_factory=FakeClusterGatewayFactory(),
        history=FakeHistory(),
        confirmations=InMemoryConfirmationStore(ttl_seconds=600),
        structures=FakeStructureBuilder(),
        job_tracker=FakeJobTracker(),
        calc_inputs=FakeCalcInputGenerator(),
        potcar_dir="/potcars",
        remote_base="becario_runs",
        calc_runs=FakeCalcRuns(),
    )
    return service, router


class Espia:
    """Doble que anota cada llamada y devuelve lo que se le diga."""

    def __init__(self, devuelve=None):
        self.devuelve = devuelve
        self.llamadas: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.llamadas.append((args, kwargs))
        return self.devuelve


# ---------------------------------------------------------------------------
# (a) Re-exports y alias
# ---------------------------------------------------------------------------


class TestReExports:
    """Tests y presentación importan estos nombres desde `services`; si el
    re-export pasara a ser una copia, monkeypatchear el original dejaría de
    alcanzar a quien lo usa (o al revés), sin que ningún import falle."""

    @pytest.mark.parametrize("nombre", [
        "ALREADY_MODIFYING_TEXT",
        "EXPIRED_CONFIRMATION_TEXT",
        "FOREIGN_CONFIRMATION_TEXT",
        "HELP_TEXT",
        "NOT_REGISTERED_TEXT",
        "ROUTER_ERROR_TEXT",
        "ROUTER_TIMEOUT_TEXT",
        "ROUTER_UNREACHABLE_TEXT",
        "USED_CONFIRMATION_TEXT",
        "_router_failure_text",
    ])
    def test_los_textos_son_los_de_textos(self, nombre):
        assert getattr(services, nombre) is getattr(textos, nombre)

    @pytest.mark.parametrize("nombre", [
        "_CANCEL_RE",
        "_InMemoryPendingEdits",
        "_is_cancel_message",
    ])
    def test_lo_de_pendientes_es_lo_de_pending_edits(self, nombre):
        assert getattr(services, nombre) is getattr(pending_edits, nombre)

    @pytest.mark.parametrize("nombre", [
        "_LISTING_MAX_CHARS",
        "_VIEW_FILE_MAX_BYTES",
        "_truncate_listing",
    ])
    def test_lo_de_archivos_remotos_es_lo_de_remote_files(self, nombre):
        assert getattr(services, nombre) is getattr(remote_files, nombre)

    def test_pending_edit_privado_es_el_del_dominio(self):
        assert services._PendingEdit is PendingEdit


class TestAliasDePlanDescriber:
    @pytest.mark.parametrize("alias, funcion", [
        ("_needs_decomposition", plan_describer.needs_decomposition),
        ("_describe_structure_step", plan_describer.describe_structure_step),
        ("_plan_steps_as_requests", plan_describer.plan_steps_as_requests),
        ("_propagate_answer", plan_describer.propagate_answer),
        ("_render_plan_context", plan_describer.render_plan_context),
    ])
    def test_el_alias_estatico_es_la_funcion_del_modulo(self, alias, funcion):
        # Desde la clase y desde una instancia: `staticmethod` no liga
        # `self`, así que las dos rutas tienen que dar la misma función.
        assert getattr(BecarioService, alias) is funcion
        assert isinstance(BecarioService.__dict__[alias], staticmethod)

    @pytest.mark.parametrize("metodo, funcion, constante", [
        ("_looks_compound", "looks_compound", "_COMPOUND_MARKERS"),
        ("_is_batch_plan", "is_batch_plan", "_MATERIALIZABLE_STEP_INTENTS"),
        ("_describe_batch_step", "describe_batch_step", "_describe_structure_step"),
    ])
    def test_el_classmethod_pasa_la_constante_de_la_clase(
        self, monkeypatch, metodo, funcion, constante
    ):
        espia = Espia(devuelve=object())
        monkeypatch.setattr(plan_describer, funcion, espia)
        arg = object()

        resultado = getattr(BecarioService, metodo)(arg)

        assert resultado is espia.devuelve
        assert espia.llamadas == [((arg, getattr(BecarioService, constante)), {})]

    def test_la_constante_la_toma_de_la_subclase(self, monkeypatch):
        # Es `cls.`, no `BecarioService.`: una subclase que cambie los
        # marcadores cambia el resultado.
        class ConOtrosMarcadores(BecarioService):
            _COMPOUND_MARKERS = (" ademas ",)

        # Diez palabras: el mínimo para que un marcador alcance.
        con_y = "mostrame el estado de mis trabajos y listame mis archivos"
        con_ademas = "mostrame el estado de mis trabajos ademas listame mis archivos"
        assert BecarioService._looks_compound(con_y)
        assert not ConOtrosMarcadores._looks_compound(con_y)
        assert ConOtrosMarcadores._looks_compound(con_ademas)

    def test_mismo_resultado_que_la_funcion_con_las_constantes(self):
        for texto in (
            "mostrame el estado de mis trabajos y listame mis archivos",
            "estado",
            "listá la carpeta de corridas; después mirá el OUTCAR del último",
        ):
            assert BecarioService._looks_compound(texto) == plan_describer.looks_compound(
                texto, BecarioService._COMPOUND_MARKERS
            )


# ---------------------------------------------------------------------------
# (b) Delegadores finos
# ---------------------------------------------------------------------------

# (método en la clase, función en plan_coordinator, cantidad de args
# posicionales, kwargs). Sale de leer `services.py`, no de adivinar.
_DELEGADORES = [
    ("_prepare_batch", "prepare_batch", 2, {}),
    ("_materialize_step_fn", "materialize_step_fn", 4, {}),
    ("_run_composite_plan", "run_composite_plan", 2, {}),
    ("_prepare_destructive_tail", "prepare_destructive_tail", 4, {}),
    ("_pop_pending_edit", "pop_pending_edit", 1, {}),
    ("_expired_pending_text", "expired_pending_text", 1, {}),
    ("sweep_expired_pendings", "sweep_expired_pendings", 0, {}),
    ("_purgar_confirmaciones_vencidas", "purgar_confirmaciones_vencidas", 0, {}),
    ("_arm_pending_edit", "arm_pending_edit", 4, {}),
    ("_arm_if_awaiting", "arm_if_awaiting", 4, {}),
    ("_apply_edit", "apply_edit", 3, {}),
    ("_router_failure_con_plan_vivo", "router_failure_con_plan_vivo", 3,
     {"awaiting_index": 2}),
    ("_fill_awaiting_step", "fill_awaiting_step", 3, {}),
    ("_execute_batch", "execute_batch", 4, {}),
    ("_execute_batch_step", "execute_batch_step", 2, {}),
]

# Los que en la clase no devuelven nada (`-> None`): delegan y descartan.
_SIN_RESULTADO = {
    "_purgar_confirmaciones_vencidas", "_arm_pending_edit", "_arm_if_awaiting",
}


class TestDelegadores:
    def test_la_tabla_cubre_todo_plan_coordinator(self):
        # Una función nueva en el coordinador sin su fila acá es un
        # delegador sin contrato: que falle este test lo hace visible.
        publicas = {
            nombre
            for nombre, f in inspect.getmembers(plan_coordinator, inspect.isfunction)
            if f.__module__ == plan_coordinator.__name__ and not nombre.startswith("_")
        }
        assert publicas == {funcion for _, funcion, _, _ in _DELEGADORES}

    @pytest.mark.parametrize(
        "metodo, funcion, n_args, kwargs", _DELEGADORES, ids=[d[0] for d in _DELEGADORES]
    )
    def test_delega_con_svc_primero_y_mismos_argumentos(
        self, servicio, monkeypatch, metodo, funcion, n_args, kwargs
    ):
        service, _ = servicio
        espia = Espia(devuelve=object())
        monkeypatch.setattr(plan_coordinator, funcion, espia)
        args = tuple(object() for _ in range(n_args))

        resultado = getattr(service, metodo)(*args, **kwargs)

        assert espia.llamadas == [((service, *args), kwargs)]
        if metodo in _SIN_RESULTADO:
            assert resultado is None
        else:
            assert resultado is espia.devuelve

    def test_argumentos_opcionales_por_defecto(self, servicio, monkeypatch):
        # Los defaults viven en la firma del delegador: si se pierden, el
        # coordinador recibe otra cosa sin que nadie lo note.
        service, _ = servicio
        espias = {
            f: Espia() for f in (
                "materialize_step_fn", "arm_pending_edit", "router_failure_con_plan_vivo",
            )
        }
        for f, espia in espias.items():
            monkeypatch.setattr(plan_coordinator, f, espia)
        ctx, step, edit, exc = object(), object(), object(), object()

        service._materialize_step_fn(ctx, step)
        service._arm_pending_edit(1, 2, [])
        service._router_failure_con_plan_vivo(ctx, edit, exc)

        assert espias["materialize_step_fn"].llamadas == [((service, ctx, step, 1, None), {})]
        assert espias["arm_pending_edit"].llamadas == [((service, 1, 2, [], None), {})]
        assert espias["router_failure_con_plan_vivo"].llamadas == [
            ((service, ctx, edit, exc), {"awaiting_index": None})
        ]


# ---------------------------------------------------------------------------
# (c) Ruteo por `svc.`
# ---------------------------------------------------------------------------

_DOS_PASOS = [
    (Intent.CHECK_STATUS, {}),
    (Intent.LIST_FILES, {"ruta": "/home/alice/a"}),
]


class TestRuteoPorSvc:
    """Se maneja el flujo real (`handle_text`) y se parchea UN método en la
    instancia: si el coordinador lo llamara por módulo en vez de por
    `svc.`, el parche no se vería y el test fallaría."""

    def test_la_edicion_de_un_plan_vuelve_a_despachar_por_svc(
        self, servicio, monkeypatch
    ):
        service, router = servicio
        despacho = Espia(devuelve=Reply(text="despachado"))
        monkeypatch.setattr(service, "_dispatch_plan", despacho)
        service._arm_pending_edit(ALICE.telegram_user_id, 1, list(_DOS_PASOS))
        router.next_edit = (2, {"ruta": "/home/alice/b"})

        reply = service.handle_text(
            chat_id=1, user_id=ALICE.telegram_user_id, text="paso 2: mirá /b"
        )

        assert reply is despacho.devuelve
        [((ctx, plan), _)] = despacho.llamadas
        assert ctx.user_id == ALICE.telegram_user_id
        assert isinstance(plan, Plan)
        assert [(s.action, s.parametros) for s in plan.steps] == [
            (Intent.CHECK_STATUS, {}),
            (Intent.LIST_FILES, {"ruta": "/home/alice/b"}),
        ]

    def test_completar_un_dato_faltante_vuelve_a_despachar_por_svc(
        self, servicio, monkeypatch
    ):
        service, router = servicio
        despacho = Espia(devuelve=Reply(text="despachado"))
        monkeypatch.setattr(service, "_dispatch_plan", despacho)
        service._arm_pending_edit(
            ALICE.telegram_user_id, 1, list(_DOS_PASOS), awaiting_index=2
        )
        router.next = RoutedRequest(
            intent=Intent.LIST_FILES, params={"ruta": "/home/alice/c"}
        )

        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="/c")

        assert reply is despacho.devuelve
        [((_, plan), _)] = despacho.llamadas
        assert plan.steps[1].parametros == {"ruta": "/home/alice/c"}

    def test_un_cambio_ambiguo_rearma_el_plan_por_svc(self, servicio, monkeypatch):
        service, router = servicio
        service._arm_pending_edit(ALICE.telegram_user_id, 1, list(_DOS_PASOS))
        rearmado = Espia()
        monkeypatch.setattr(service, "_arm_pending_edit", rearmado)
        router.next_edit = (None, {"ruta": "/x"})

        service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="cambiá eso")

        assert rearmado.llamadas == [((ALICE.telegram_user_id, 1, _DOS_PASOS), {})]

    def test_un_router_caido_editando_pasa_por_svc(self, servicio, monkeypatch):
        service, router = servicio
        service._arm_pending_edit(ALICE.telegram_user_id, 1, list(_DOS_PASOS))
        exc = RouterUnavailableError(RouterFailureReason.UNREACHABLE, "caído")

        def extract_edit(plan_context, text):
            raise exc

        monkeypatch.setattr(router, "extract_edit", extract_edit)
        fallo = Espia(devuelve=Reply(text="sigue esperando", ok=False))
        monkeypatch.setattr(service, "_router_failure_con_plan_vivo", fallo)

        reply = service.handle_text(chat_id=1, user_id=ALICE.telegram_user_id, text="x")

        assert reply is fallo.devuelve
        [((_, edit, recibida), _)] = fallo.llamadas
        assert recibida is exc
        assert edit.steps == _DOS_PASOS

    def test_el_barrido_usa_el_aviso_y_la_purga_de_la_instancia(
        self, servicio, monkeypatch
    ):
        service, _ = servicio
        service._pending_edits.put(
            ALICE.telegram_user_id,
            PendingEdit(steps=list(_DOS_PASOS), created_at=0.0, chat_id=7),
        )
        aviso = Espia(devuelve="aviso parcheado")
        purga = Espia()
        monkeypatch.setattr(service, "_expired_pending_text", aviso)
        monkeypatch.setattr(service, "_purgar_confirmaciones_vencidas", purga)

        avisos = service.sweep_expired_pendings()

        assert avisos == [(7, "aviso parcheado")]
        assert len(aviso.llamadas) == 1
        assert purga.llamadas == [((), {})]
