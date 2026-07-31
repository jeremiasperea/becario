"""Tests del ASEStructureBuilder con ASE real (sin red, sin cluster)."""
import pytest
from pydantic import ValidationError

from becario.domain.models import (
    Axis,
    OutputFormat,
    StructureKind,
    StructureRequest,
)
from becario.infrastructure.ase_builder import (
    ASEStructureBuilder,
    StructureBuildError,
    build_structure_atoms,
)


@pytest.fixture()
def builder(tmp_path):
    return ASEStructureBuilder(workdir=str(tmp_path))


def _slab_req(miller, layers=None, supercell=(1, 1, 1), axis=Axis.Z):
    """Losa de ZrO2 fluorita: compuesto cúbico, el caso donde la celda
    primitiva de ASE vuelve ambiguos los índices de Miller."""
    return StructureRequest(
        formula="ZrO2",
        kind=StructureKind.SLAB,
        crystal="fluorite",
        lattice_a=5.14,
        miller=miller,
        layers=layers,
        supercell=supercell,
        vacuum_axis=axis,
    )


def _slab_atoms(**kwargs):
    req = _slab_req(**kwargs)
    return build_structure_atoms(
        req.kind, req.formula, req.crystal, req.lattice_a,
        miller=req.miller, layers=req.layers, vacuum=req.vacuum,
        vacuum_axis=req.vacuum_axis, supercell=req.supercell,
    )


class TestBulk:
    def test_silicon_diamond_primitive(self, builder):
        res = builder.build(StructureRequest(formula="Si"))
        assert res.chemical_formula == "Si2"  # celda primitiva de diamante
        assert res.n_atoms == 2
        assert res.filename.startswith("POSCAR_Si")

    def test_supercell_multiplies_atoms(self, builder):
        res = builder.build(StructureRequest(formula="Si", supercell=(2, 2, 2)))
        assert res.n_atoms == 16  # 2 átomos * 8

    def test_compound_with_crystal_and_lattice(self, builder):
        res = builder.build(
            StructureRequest(formula="NaCl", crystal="rocksalt", lattice_a=5.64)
        )
        assert res.n_atoms == 2  # celda primitiva rocksalt: 1 Na + 1 Cl
        assert "Na" in res.chemical_formula and "Cl" in res.chemical_formula

    def test_compound_without_parameters_gives_friendly_error(self, builder):
        with pytest.raises(StructureBuildError, match="red cristalina"):
            builder.build(StructureRequest(formula="TiO2"))

    def test_poscar_file_is_valid_and_readable(self, builder, tmp_path):
        from ase.io import read

        res = builder.build(StructureRequest(formula="Cu", supercell=(2, 1, 1)))
        atoms = read(res.local_path, format="vasp")
        assert len(atoms) == res.n_atoms
        assert atoms.get_chemical_formula() == res.chemical_formula


class TestMolecule:
    def test_water_from_g2(self, builder):
        res = builder.build(
            StructureRequest(formula="H2O", kind=StructureKind.MOLECULE)
        )
        assert res.n_atoms == 3
        assert res.chemical_formula == "H2O"

    def test_molecule_gets_vacuum_cell(self, builder):
        # Sin celda no se puede escribir POSCAR: el builder centra con vacío.
        res = builder.build(
            StructureRequest(formula="CO2", kind=StructureKind.MOLECULE, vacuum=12.0)
        )
        assert res.n_atoms == 3
        assert "a=" in res.cell_summary

    def test_unknown_molecule_friendly_error(self, builder):
        # Fe2: elementos reales (pasa la validación de fórmula), pero no es
        # una molécula de la base G2 de ASE, así que el builder debe traducir
        # el fallo interno a un StructureBuildError amistoso.
        with pytest.raises(StructureBuildError, match="G2"):
            builder.build(
                StructureRequest(formula="Fe2", kind=StructureKind.MOLECULE)
            )


class TestOutputFormats:
    @pytest.mark.parametrize("fmt,ext", [(OutputFormat.CIF, ".cif"), (OutputFormat.XYZ, ".xyz")])
    def test_alternative_formats(self, builder, fmt, ext):
        res = builder.build(StructureRequest(formula="Si", output_format=fmt))
        assert res.filename.endswith(ext)


class TestStructureRequestValidation:
    @pytest.mark.parametrize("evil", ["Si; rm -rf /", "../etc", "", "Si Cl", "S\ni"])
    def test_formula_injection_rejected(self, evil):
        with pytest.raises(ValidationError):
            StructureRequest(formula=evil)

    def test_formula_whitespace_normalized(self):
        assert StructureRequest(formula="Si\n").formula == "Si"

    def test_supercell_bounds(self):
        with pytest.raises(ValidationError):
            StructureRequest(formula="Si", supercell=(0, 1, 1))
        with pytest.raises(ValidationError):
            StructureRequest(formula="Si", supercell=(11, 1, 1))

    def test_remote_dest_dir_validation(self):
        with pytest.raises(ValidationError):
            StructureRequest(formula="Si", remote_dest_dir="relativo/mal")
        with pytest.raises(ValidationError):
            StructureRequest(formula="Si", remote_dest_dir="/home/../etc")
        ok = StructureRequest(formula="Si", remote_dest_dir="/home/user/calc")
        assert ok.remote_dest_dir == "/home/user/calc"

    def test_lattice_bounds(self):
        with pytest.raises(ValidationError):
            StructureRequest(formula="Si", lattice_a=0.1)
        with pytest.raises(ValidationError):
            StructureRequest(formula="Si", lattice_a=100.0)


class TestSlab:
    """La losa se corta con `ase.build.surface` sobre la celda CONVENCIONAL.
    Todo lo que se verifica acá es física, no formato: si estos tests pasan
    pero la superficie es otra, el POSCAR igual se escribe perfecto."""

    def test_conventional_cell_makes_miller_indices_mean_what_they_say(self, builder):
        """Sobre la celda PRIMITIVA de fluorita, (001) y (111) dan la misma
        superficie: ASE lee los índices en la base de la celda que recibe.
        Con la convencional son distintas — que es la razón de `cubic=True`."""
        res001 = builder.build(_slab_req(miller=(0, 0, 1)))
        res111 = builder.build(_slab_req(miller=(1, 1, 1)))
        assert res001.cell_summary != res111.cell_summary

    def test_vacuum_defaults_to_15_angstrom_on_z(self, builder):
        res = builder.build(_slab_req(miller=(0, 0, 1)))
        assert "15 Å de vacío en z" in res.slab_summary

    def test_vacuum_only_on_one_axis(self, builder):
        """La losa NO pasa por `center()`, que infla los tres ejes: las dos
        direcciones del plano quedan con el tamaño de la celda cortada."""
        atoms = _slab_atoms(miller=(0, 0, 1))
        a, b, c = atoms.cell.lengths()
        assert a < 10 and b < 10  # plano: parámetros de red, sin vacío
        assert c > 30             # normal: espesor + 2 x 15 Å

    def test_lying_slab_is_the_same_structure_rotated(self):
        """Acostar la losa es una rotación rígida: mismas distancias
        interatómicas, mismo volumen, y celda DERECHA (det > 0 — un
        intercambio de dos ejes la dejaría zurda, o sea volumen negativo
        en el POSCAR)."""
        import numpy as np

        standing = _slab_atoms(miller=(1, 1, 0), axis=Axis.Z)
        for axis in (Axis.X, Axis.Y):
            lying = _slab_atoms(miller=(1, 1, 0), axis=axis)
            assert np.allclose(
                np.sort(standing.get_all_distances(mic=True).ravel()),
                np.sort(lying.get_all_distances(mic=True).ravel()),
            )
            assert np.isclose(standing.get_volume(), lying.get_volume())
            assert np.linalg.det(lying.cell.array) > 0

    def test_lying_slab_puts_vacuum_on_requested_axis(self):
        """Se mide en CARTESIANAS, no con `cell.lengths()`: la permutación
        reparte las componentes de los vectores de red, así que el largo del
        vector `c` ya no dice sobre qué eje cae el vacío."""
        import numpy as np

        for axis in (Axis.X, Axis.Y, Axis.Z):
            atoms = _slab_atoms(miller=(1, 1, 0), axis=axis)
            extent = np.abs(atoms.cell.array).sum(axis=0)  # celda, por eje
            span = np.ptp(atoms.positions, axis=0)         # átomos, por eje
            free = extent - span
            assert int(np.argmax(free)) == axis.index
            assert free[axis.index] > 25  # 2 x 15 Å menos una capa

    def test_in_plane_supercell_repeats_only_the_plane(self, builder):
        one = _slab_atoms(miller=(0, 0, 1))
        two = _slab_atoms(miller=(0, 0, 1), supercell=(2, 1, 1))
        assert len(two) == 2 * len(one)
        assert two.cell.lengths()[2] == pytest.approx(one.cell.lengths()[2])

    def test_hcp_falls_back_when_no_cubic_cell_exists(self, builder):
        """`cubic=True` no existe para hcp; ahí la celda default de ASE ya es
        la convencional, así que la losa se corta igual en vez de fallar."""
        res = builder.build(
            StructureRequest(
                formula="Zr", kind=StructureKind.SLAB, crystal="hcp",
                lattice_a=3.23, miller=(0, 0, 1), layers=4,
            )
        )
        assert res.n_atoms > 0


class TestSlabValidation:
    def test_miller_is_required_never_guessed(self):
        """Elegir la cara equivocada cambia la física: no se adivina."""
        with pytest.raises(ValidationError, match="Miller"):
            StructureRequest(formula="ZrO2", kind=StructureKind.SLAB)

    def test_zero_miller_rejected(self):
        with pytest.raises(ValidationError, match="no define un plano"):
            StructureRequest(
                formula="ZrO2", kind=StructureKind.SLAB, miller=(0, 0, 0)
            )

    def test_slab_supercell_cannot_repeat_along_normal(self):
        """Repetir en la normal duplicaría la losa Y su vacío."""
        with pytest.raises(ValidationError, match="tercera"):
            _slab_req(miller=(0, 0, 1), supercell=(2, 1, 2))

    def test_bulk_rejects_slab_fields_instead_of_ignoring_them(self):
        """Si un pedido de bulk trae miller, algo se armó mal: se rechaza en
        vez de descartarlo en silencio."""
        with pytest.raises(ValidationError, match="solo aplican a una losa"):
            StructureRequest(formula="Si", miller=(0, 0, 1))

    def test_defaults_resolved_into_the_request(self):
        req = _slab_req(miller=(1, 1, 1))
        assert req.vacuum == 15.0
        assert req.layers == 5
