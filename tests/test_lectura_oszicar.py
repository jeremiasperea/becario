"""El OSZICAR se lee acotado y se avisa si quedó cortado."""

from typing import Optional

from becario.application.lectura_oszicar import (
    OSZICAR_MAX_BYTES,
    leer_oszicar,
)


class _Cluster:
    def __init__(self, files: dict[str, str]):
        self.files = files
        self.max_bytes: list[Optional[int]] = []

    def read_file(self, path: str, max_bytes: Optional[int] = None) -> Optional[str]:
        self.max_bytes.append(max_bytes)
        texto = self.files.get(path)
        if texto is not None and max_bytes is not None:
            texto = texto[:max_bytes]
        return texto


def test_lee_con_el_tope():
    cluster = _Cluster({"/r/OSZICAR": "E0= -1.0\n"})
    lectura = leer_oszicar(cluster, "/r")
    assert cluster.max_bytes == [OSZICAR_MAX_BYTES]
    assert lectura.texto == "E0= -1.0\n" and not lectura.truncado


def test_al_tope_es_truncado():
    cluster = _Cluster({"/r/OSZICAR": "x" * (OSZICAR_MAX_BYTES + 10)})
    assert leer_oszicar(cluster, "/r").truncado


def test_falta_no_es_truncado():
    lectura = leer_oszicar(_Cluster({}), "/r")
    assert lectura.texto is None and not lectura.truncado
