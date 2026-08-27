"""El vocabulario de datos de una corrida, y los parseos que lo sostienen.

Sale de medir E6: la pregunta real —«¿cuántas vueltas iónicas hizo?»— no
pedía que el bot interpretara un archivo, pedía un conteo que el repo YA
hacía en `relaxed_source._check_convergence` para avisar si una relajación
se había quedado sin NSW.

El fragmento de OSZICAR con el que se prueba es el REAL de esa corrida (job
14, Zr_relajacion), tal como el bot lo mostró en el chat: truncado. Se usa
así a propósito — es lo que el usuario tenía delante cuando preguntó.
"""
from pathlib import Path

import pytest

from becario.domain.datos_de_corrida import (
    DESCRIPCIONES,
    NINGUNO,
    DatoDeCorrida,
    contar_atomos,
    contar_pasos_ionicos,
    dato_valido,
    nsw_de,
    ultima_energia,
)

_OSZICAR_REAL = (
    Path(__file__).parent / "fixtures" / "contenido"
    / "oszicar_zr_relajacion_truncado.txt"
).read_text(encoding="utf-8")

_CONTCAR = """Zr
   1.00000000000000
     3.2415090615 0.0000000000 0.0000000000
    -1.6207545308 2.8071618608 0.0000000000
     0.0000000000 0.0000000000 5.1670000000
   Zr
     2
Direct
  0.3333333 0.6666667 0.2500000
  0.6666667 0.3333333 0.7500000
"""


class TestElVocabularioSoloPrometeLoQueSabeHacer:
    """Cada valor del enum tiene que tener quién lo calcule y cómo
    explicarlo. Un valor sin descripción llega al modelo como una palabra
    suelta; uno sin cálculo es una promesa que el handler no cumple."""

    def test_todo_valor_tiene_descripcion(self):
        faltan = [d.value for d in DatoDeCorrida if d not in DESCRIPCIONES]
        assert faltan == []

    def test_ninguno_no_es_un_valor_del_enum(self):
        # Es la AUSENCIA de dato. Si fuera un valor más, algún camino iba a
        # terminar intentando calcularlo.
        assert NINGUNO not in DatoDeCorrida._value2member_map_


class TestDatoValido:
    @pytest.mark.parametrize("crudo, esperado", [
        ("pasos_ionicos", DatoDeCorrida.PASOS_IONICOS),
        ("  ENERGIA  ", DatoDeCorrida.ENERGIA),
        ("parametros_red", DatoDeCorrida.PARAMETROS_RED),
    ])
    def test_reconoce_los_del_vocabulario(self, crudo, esperado):
        assert dato_valido(crudo) is esperado

    @pytest.mark.parametrize("crudo", [None, "", "ninguno", "diagnostico", "vueltas"])
    def test_todo_lo_demas_es_none(self, crudo):
        # Los cuatro casos de «no hay dato pedido» colapsan en uno solo: que
        # el router no preguntara, que se abstuviera, que inventara un valor
        # o que llegara vacío terminan todos en la respuesta general.
        assert dato_valido(crudo) is None


class TestContarPasosIonicos:
    def test_cuenta_las_vueltas_del_oszicar_real(self):
        # Dos líneas 'N F=' completas en el fragmento que el bot mostró.
        assert contar_pasos_ionicos(_OSZICAR_REAL) == 2

    def test_las_lineas_electronicas_no_cuentan(self):
        # DAV/RMM son pasos ELECTRÓNICOS: hay decenas por cada vuelta iónica.
        # Contarlas daría un número enorme y creíble, que es el peor error.
        assert "DAV:" in _OSZICAR_REAL and "RMM:" in _OSZICAR_REAL
        assert contar_pasos_ionicos(_OSZICAR_REAL) < _OSZICAR_REAL.count("RMM:")

    def test_un_oszicar_sin_pasos_da_cero_no_none(self):
        # Cero pasos es un HECHO (un single point, o murió antes de la
        # primera vuelta); no hay archivo es otra cosa.
        assert contar_pasos_ionicos("N       E\nDAV:   1  0.9E+02\n") == 0

    def test_sin_archivo_es_none(self):
        assert contar_pasos_ionicos(None) is None
        assert contar_pasos_ionicos("") is None


class TestUltimaEnergia:
    def test_toma_el_ultimo_e0_no_el_primero(self):
        # El OSZICAR crece hacia abajo: la energía que vale es la última.
        assert ultima_energia(_OSZICAR_REAL) == pytest.approx(-17.097775)

    def test_sin_e0_es_none(self):
        assert ultima_energia("N       E\nDAV:   1  0.9E+02\n") is None

    def test_sin_archivo_es_none(self):
        assert ultima_energia(None) is None


class TestNsw:
    def test_lo_lee_del_incar(self):
        assert nsw_de("IBRION = 2\nNSW = 180\nENCUT = 400\n") == 180

    def test_no_le_importan_mayusculas_ni_espacios(self):
        assert nsw_de("  nsw   =   60\n") == 60

    def test_sin_nsw_es_none(self):
        assert nsw_de("IBRION = 2\n") is None
        assert nsw_de(None) is None


class TestContarAtomos:
    def test_suma_la_linea_de_cantidades(self):
        assert contar_atomos(_CONTCAR) == 2

    def test_soporta_poscar_sin_linea_de_simbolos(self):
        # VASP 4 no trae la 6ª línea con los símbolos: las cantidades suben
        # una posición, y por eso se prueban los dos índices.
        sin_simbolos = _CONTCAR.replace("   Zr\n     2\n", "     2\n")
        assert contar_atomos(sin_simbolos) == 2

    def test_varios_elementos_se_suman(self):
        multi = _CONTCAR.replace("   Zr\n     2\n", "   Zr O\n     4 8\n")
        assert contar_atomos(multi) == 12

    def test_sin_archivo_es_none(self):
        assert contar_atomos(None) is None
