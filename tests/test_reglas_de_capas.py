"""Regla de dependencias de la arquitectura: las capas internas no conocen a las externas.

`domain` no importa nada de `application`, `infrastructure` ni `presentation`;
`application` no importa `infrastructure` ni `presentation`. Las
implementaciones concretas entran solo por el composition root (`main.py`).
Hoy se cumple; este test es lo que impide que deje de cumplirse en silencio.

Lee el código con `ast` en vez de importarlo: cubre también los imports bajo
`TYPE_CHECKING` y los que están adentro de funciones.
"""
import ast
from pathlib import Path

import pytest

PAQUETE = Path(__file__).resolve().parent.parent / "becario"

PROHIBIDAS = {
    "domain": ("application", "infrastructure", "presentation"),
    "application": ("infrastructure", "presentation"),
}


def _modulo(path: Path, raiz: Path) -> str:
    partes = path.relative_to(raiz.parent).with_suffix("").parts
    return ".".join(partes[:-1] if partes[-1] == "__init__" else partes)


def _imports(path: Path, raiz: Path = PAQUETE) -> list[tuple[int, str]]:
    """Módulos absolutos que importa el archivo, con relativos ya resueltos."""
    modulo = _modulo(path, raiz)
    paquete = modulo if path.name == "__init__.py" else modulo.rpartition(".")[0]
    encontrados = []
    for nodo in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Import):
            encontrados += [(nodo.lineno, alias.name) for alias in nodo.names]
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.level:
                base = paquete.split(".")
                base = base[: len(base) - (nodo.level - 1)]
                destino = ".".join(base + ([nodo.module] if nodo.module else []))
            else:
                destino = nodo.module or ""
            encontrados.append((nodo.lineno, destino))
            # `from .. import infrastructure` nombra la capa en el alias.
            encontrados += [
                (nodo.lineno, f"{destino}.{alias.name}") for alias in nodo.names
            ]
    return encontrados


@pytest.mark.parametrize("capa", sorted(PROHIBIDAS))
def test_la_capa_no_importa_capas_externas(capa):
    prohibidos = tuple(f"becario.{p}" for p in PROHIBIDAS[capa])
    violaciones = [
        f"{path.relative_to(PAQUETE.parent)}:{linea} importa {destino}"
        for path in sorted((PAQUETE / capa).rglob("*.py"))
        for linea, destino in _imports(path)
        if destino in prohibidos or destino.startswith(tuple(p + "." for p in prohibidos))
    ]
    assert not violaciones, "\n".join(violaciones)


def test_resuelve_imports_relativos_hacia_otra_capa(tmp_path):
    """Sin esto, un `from ..infrastructure import x` pasaría desapercibido."""
    falso = tmp_path / "becario"
    (falso / "application" / "handlers").mkdir(parents=True)
    archivo = falso / "application" / "handlers" / "x.py"
    archivo.write_text(
        "from ...infrastructure.ssh_gateway import G\n"
        "from ... import presentation\n"
        "from .calc import algo\n"
    )

    destinos = [d for _, d in _imports(archivo, raiz=falso)]

    assert "becario.infrastructure.ssh_gateway" in destinos
    assert "becario.presentation" in destinos
    assert "becario.application.handlers.calc" in destinos
