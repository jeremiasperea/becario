"""Lo que el bot NO sabe hacer, reconocido antes de intentarlo.

El caso que lo motiva salió de usar el bot: un pedido de apilar Zr sobre W
volvió como un batch de ocho pasos listo para aprobar de un botón. El
pedido era imposible desde el principio.

Este archivo tiene dos mitades y ninguna sobra. La primera fija QUÉ se
reconoce, y sobre todo qué NO: un falso positivo bloquea un pedido
legítimo, que es lo peor que puede pasar acá. La segunda fija que el corte
pasa ANTES del router — mirando el plan emitido no hay nada que objetar,
porque el modelo emite pasos de Zr y de W por separado, cada uno válido.
"""
import pytest

from becario.domain.fuera_de_alcance import (
    LO_QUE_SE_ARMAR,
    NO_SE_APILAR,
    fuera_de_alcance,
)


class TestStackedMaterialsAreRecognized:
    @pytest.mark.parametrize("texto", [
        # El mensaje real, tal como se escribió (con su typo incluido).
        "Arma una supercelda de Zr sobre W 2x2/3x3 respectivamente, "
        "no olvides agregas 15 ang de vacío en z",
        "armá Zr sobre W",
        "poné W encima de Si",
        "quiero una heteroestructura de Zr y W",
        "hacé una heterostructura Zr/W",
        "armame una bicapa",
        "necesito un slab de Zr sobre un sustrato de W",
        "apilá dos capas de grafeno",
        "adsorción de H sobre Ni",
    ])
    def test_it_bounces(self, texto):
        assert fuera_de_alcance(texto) == NO_SE_APILAR

    def test_the_message_says_what_can_be_done(self):
        # Sin esto un falso positivo deja al usuario sin salida: le dijimos
        # que no y no le dijimos por dónde sí.
        assert LO_QUE_SE_ARMAR in NO_SE_APILAR
        assert "POSCAR" in NO_SE_APILAR and "slab" in NO_SE_APILAR

    def test_accents_and_dictation_do_not_matter(self):
        # El pedido llega dictado por voz tanto como escrito.
        assert fuera_de_alcance("apila Zr y W") == fuera_de_alcance("apilá Zr y W")


class TestLegitimateRequestsAreLeftAlone:
    """La parte que importa. `sobre` es una preposición común y `interfaz`
    en castellano es también la de usuario: un chequeo goloso convierte
    pedidos buenos en un «no sé hacer eso» sin salida."""

    @pytest.mark.parametrize("texto", [
        "relajá el bulk de Zr hcp",
        "Calcula el bulk de W bcc",
        "generá un POSCAR de Si diamond 2x2x2",
        "armá el slab de ZrO2 (001) con 5 capas",
        "mostrame el CONTCAR",
        "dame información sobre el cluster",
        "contame sobre el último cálculo",
        "qué archivos hay en /data/becario_runs",
        "acá debería haber una opción de modificar la interfaz",
        "hacé la curva de convergencia de ENCUT para Zr hcp",
        "cancelá el trabajo 12345",
        "creame la carpeta pilas en mis corridas",
        "",
    ])
    def test_it_does_not_bounce(self, texto):
        assert fuera_de_alcance(texto) is None

    def test_two_words_around_sobre_are_not_two_materials(self):
        # El lado izquierdo y el derecho tienen que ser fórmulas de verdad.
        assert fuera_de_alcance("Mostrame lo que dice el manual sobre ISMEAR") is None

    def test_a_calculation_of_two_materials_is_still_two_calculations(self):
        # "relajá W y Si" es un pedido compuesto legítimo, no un apilado.
        assert fuera_de_alcance("relajá el bulk de W y el de Si") is None
