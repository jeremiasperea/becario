"""Tests de `domain/sugerencias.py`: qué le falta a un material.

Todo puro y sin I/O: las reglas reciben corridas ya traducidas. Lo que se
prueba acá no es el texto —eso cambia— sino las tres promesas del módulo:
que solo sugiere lo que el bot sabe preparar, que no encadena nada sobre
una corrida que terminó mal, y que las citas del manual salen del
vocabulario y no de la imaginación de quien escribió la regla.
"""
import pytest

from becario.domain.models import CalcKind
from becario.domain.sugerencias import CorridaPrevia, Sugerencia, render, sugerencias


def _c(kind: CalcKind, *, estado: str = "completado", encut=520, formula="Zr"):
    return CorridaPrevia(
        formula=formula, calc_kind=kind, fecha="2026-08-01",
        estado=estado, encut=encut, job_name=f"{formula}_{kind.value}",
    )


class TestNadaQueDecir:
    def test_sin_corridas_no_hay_sugerencias(self):
        assert sugerencias([]) == []

    def test_con_la_secuencia_completa_no_molesta(self):
        completa = [_c(CalcKind.ENCUT_SCAN), _c(CalcKind.RELAX), _c(CalcKind.STATIC)]
        assert sugerencias(completa) == []


class TestLaCorridaQueTerminoMal:
    @pytest.mark.parametrize("estado", ["falló", "cancelado", "sin rastro", "expiró"])
    def test_se_avisa_y_no_se_encadena_nada_encima(self, estado):
        """La regla que va primera, y la que más importa: sugerir el paso
        siguiente sobre una base que no existe es peor que callarse."""
        fuera = sugerencias([_c(CalcKind.RELAX, estado=estado)])

        assert len(fuera) == 1
        assert estado in fuera[0].texto
        assert fuera[0].pedido is None, "ofreció encadenar sobre una corrida rota"

    def test_un_trabajo_en_vuelo_no_cuenta_como_roto(self):
        # 'enviado' es un trabajo que todavía puede terminar bien.
        fuera = sugerencias([_c(CalcKind.RELAX, estado="enviado")])
        assert all("no sirve como punto de partida" not in s.texto for s in fuera)


class TestEncutSinBarrido:
    def test_se_sugiere_el_barrido_y_se_cita_el_manual(self):
        fuera = sugerencias([_c(CalcKind.RELAX)])
        encut = next(s for s in fuera if "ENCUT" in s.texto)

        assert "§6.9" in encut.texto, "la cita de ENCUT no salió del vocabulario"
        assert encut.pedido == "hacé la curva de convergencia de ENCUT para Zr"

    def test_con_barrido_hecho_no_se_repite(self):
        fuera = sugerencias([_c(CalcKind.ENCUT_SCAN), _c(CalcKind.RELAX)])
        assert all("barrido de convergencia" not in s.texto for s in fuera)

    def test_sin_encut_registrado_no_se_inventa_el_aviso(self):
        # Huellas viejas pueden no tener ENCUT: mejor callarse que suponer.
        fuera = sugerencias([_c(CalcKind.RELAX, encut=None)])
        assert all("ENCUT=" not in s.texto for s in fuera)


class TestRelajacionSinEstatico:
    def test_se_sugiere_cerrar_con_el_estatico(self):
        fuera = sugerencias([_c(CalcKind.RELAX)])
        est = next(s for s in fuera if "estático" in s.texto)
        assert est.pedido == "hacé un estático del Zr relajado"

    def test_una_relajacion_fallida_no_habilita_la_sugerencia(self):
        """El estático parte de la geometría relajada: si la relajación
        falló, no hay geometría de la cual partir."""
        fuera = sugerencias([_c(CalcKind.RELAX, estado="falló")])
        assert all("estático" not in (s.pedido or "") for s in fuera)


class TestDosSinRelajar:
    def test_se_avisa_que_la_geometria_no_es_la_de_equilibrio(self):
        fuera = sugerencias([_c(CalcKind.DOS)])
        dos = next(s for s in fuera if "DOS" in s.texto)
        assert dos.pedido == "relajá el Zr"

    def test_con_relajacion_previa_no_se_avisa(self):
        fuera = sugerencias([_c(CalcKind.DOS), _c(CalcKind.RELAX)])
        assert all("densidad de estados" not in s.texto for s in fuera)


class TestSoloSeSugiereLoQueElBotSabeHacer:
    def test_ningun_pedido_cae_fuera_de_CalcKind(self):
        """La promesa del módulo. Recomendar un barrido de k-points sería un
        consejo correcto y una promesa que el bot no puede cumplir: `CalcKind`
        tiene cuatro valores y las sugerencias no pueden salirse de ahí."""
        casos = [
            [_c(CalcKind.RELAX)], [_c(CalcKind.DOS)], [_c(CalcKind.STATIC)],
            [_c(CalcKind.RELAX), _c(CalcKind.DOS)],
        ]
        vocabulario = {
            "convergencia de ENCUT": CalcKind.ENCUT_SCAN,
            "estático": CalcKind.STATIC,
            "relajá": CalcKind.RELAX,
        }
        for corridas in casos:
            for s in sugerencias(corridas):
                if s.pedido:
                    assert any(k in s.pedido for k in vocabulario), (
                        f"el pedido {s.pedido!r} no corresponde a ningún CalcKind"
                    )


class TestRender:
    def test_sin_corridas_dice_por_dónde_empezar(self):
        texto = render("W", [])
        assert "No tengo ninguna corrida tuya de W" in texto

    def test_lista_los_pedidos_para_copiar(self):
        texto = render("Zr", [_c(CalcKind.RELAX)])
        assert "¿Arranco por alguno?" in texto
        assert "«hacé la curva de convergencia de ENCUT para Zr»" in texto

    def test_cuando_no_falta_nada_lo_dice_sin_inventar_trabajo(self):
        completa = [_c(CalcKind.ENCUT_SCAN), _c(CalcKind.RELAX), _c(CalcKind.STATIC)]
        texto = render("Zr", completa)
        assert "No veo nada pendiente" in texto
        assert "¿Arranco" not in texto


# ---------------------------------------------------------------------------
# El handler: traducir lo que hay en la base a lo que las reglas entienden
# ---------------------------------------------------------------------------

import json  # noqa: E402

from becario.application.handlers.queries import suggest  # noqa: E402
from becario.domain.models import Intent  # noqa: E402

from .test_service import ALICE, RoutedRequest, env  # noqa: E402,F401


def _huella(formula="Zr", calc_kind="relajacion", encut=520) -> str:
    return json.dumps({"formula": formula, "calc_kind": calc_kind, "encut": encut})


def _ctx(service):
    from becario.application.context import _Ctx
    return _Ctx(
        chat_id=1, user_id=ALICE.telegram_user_id, identity=ALICE,
        cluster=service._cluster_factory.for_identity(ALICE),
    )


class TestElHandlerArmaElContexto:
    """Ninguna de las dos tablas alcanza sola: la huella no sabe si el
    trabajo falló, y el historial no sabe con qué ENCUT se corrió."""

    def test_cruza_la_huella_con_el_estado_del_historial(self, env):
        service, _, _, history, *_ = env
        service._calc_runs.add(
            owner_id=ALICE.telegram_user_id, job_id="11", job_name="Zr_relajacion",
            fingerprint=_huella(), run_dir="/r",
        )
        # El doble de historial devuelve filas fijadas, no lo que se le
        # agregó: se fijan acá, que es como lo usan los demás tests.
        history.rows = [{"job_id": "11", "estado": "falló"}]

        reply = suggest(service, _ctx(service), {"formula": "Zr"})

        # El estado vino del historial y cambió la sugerencia entera: sobre
        # una corrida que falló no se encadena nada.
        assert "«falló»" in reply.text
        assert "convergencia de ENCUT" not in reply.text

    def test_sin_historial_la_corrida_se_toma_como_utilizable(self, env):
        service, *_ = env
        service._calc_runs.add(
            owner_id=ALICE.telegram_user_id, job_id="11", job_name="Zr_relajacion",
            fingerprint=_huella(), run_dir="/r",
        )

        reply = suggest(service, _ctx(service), {"formula": "Zr"})

        assert "convergencia de ENCUT" in reply.text

    def test_una_huella_rota_no_tira_la_consulta(self, env):
        service, *_ = env
        service._calc_runs.add(
            owner_id=ALICE.telegram_user_id, job_id="1", job_name="Zr_x",
            fingerprint="{no es json", run_dir="/r",
        )
        service._calc_runs.add(
            owner_id=ALICE.telegram_user_id, job_id="2", job_name="Zr_relajacion",
            fingerprint=_huella(), run_dir="/r",
        )

        reply = suggest(service, _ctx(service), {"formula": "Zr"})

        assert "ENCUT" in reply.text  # la corrida buena se procesó igual

    def test_sin_material_lo_pregunta_en_vez_de_adivinar(self, env):
        service, *_ = env

        reply = suggest(service, _ctx(service), {})

        assert reply.awaiting_params is True
        assert "de qué material" in reply.text

    def test_la_intencion_esta_registrada(self, env):
        service, *_ = env
        assert Intent.SUGGEST in service._intent_handlers()
