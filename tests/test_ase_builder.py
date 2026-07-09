"""Tests del ASEStructureBuilder con ASE real (sin red, sin cluster)."""
import pytest
from pydantic import ValidationError

from becario.domain.models import (
    OutputFormat,
    StructureKind,
    StructureRequest,
)
from becario.infrastructure.ase_builder import ASEStructureBuilder, StructureBuildError


@pytest.fixture()
def builder(tmp_path):
    return ASEStructureBuilder(workdir=str(tmp_path))


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
        with pytest.raises(StructureBuildError, match="G2"):
            builder.build(
                StructureRequest(formula="Xx9", kind=StructureKind.MOLECULE)
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
