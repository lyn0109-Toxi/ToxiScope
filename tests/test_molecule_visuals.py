"""Chemical bond and vector regressions for the standard RDKit depiction."""
import builtins
import re
from collections import Counter
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from presentation.molecule import SVG_NAMESPACE, render_molecule_svg


def drawing(smiles, **kwargs):
    svg = render_molecule_svg(Chem.MolFromSmiles(smiles), **kwargs)
    assert svg is not None
    return ET.fromstring(svg)


def paths(tree):
    return list(tree.iter(f"{{{SVG_NAMESPACE}}}path"))


def bond_paths(tree):
    return [path for path in paths(tree) if any(part.startswith("bond-") for part in path.get("class", "").split())]


def bond_counts(tree):
    return Counter(next(part for part in path.get("class", "").split() if part.startswith("bond-")) for path in bond_paths(tree))


@pytest.mark.parametrize("smiles,strokes", [("CC", 1), ("C=C", 2), ("C#C", 3)])
def test_carbon_bond_orders_have_one_two_and_three_distinct_vector_strokes(smiles, strokes):
    bonds = bond_paths(drawing(smiles))
    assert len(bonds) == strokes
    assert len({path.get("d") for path in bonds}) == strokes
    assert all("bond-0" in path.get("class", "") for path in bonds)


def test_nitrobenzene_aromatic_ring_is_not_a_single_bond_hexagon():
    mol = Chem.MolFromSmiles("c1ccccc1[N+](=O)[O-]")
    tree = ET.fromstring(render_molecule_svg(mol))
    aromatic_bonds = [bond.GetIdx() for bond in mol.GetBonds() if bond.GetIsAromatic()]
    counts = bond_counts(tree)
    assert len(aromatic_bonds) == 6
    assert sum(counts[f"bond-{index}"] == 2 for index in aromatic_bonds) == 3
    assert sum(counts[f"bond-{index}"] for index in aromatic_bonds) == 9
    assert sorted(bond_counts(drawing("C1CCCCC1")).values()) == [1] * 6


def test_tetrahedral_enantiomers_have_different_wedge_and_hashed_bond_geometry():
    left, right = drawing("F[C@](Cl)(Br)I"), drawing("F[C@@](Cl)(Br)I")
    left_bonds, right_bonds = bond_paths(left), bond_paths(right)
    assert [(p.get("d"), p.get("style")) for p in left_bonds] != [(p.get("d"), p.get("style")) for p in right_bonds]
    # A filled closed wedge and a sequence of short hashed strokes must be
    # distinguished in the drawing itself, independently of title/SMILES text.
    assert any("Z" in path.get("d", "") and "fill:none" not in path.get("style", "") for path in right_bonds)
    assert len(left_bonds) > len(right_bonds)


def _line_endpoints(path):
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path.get("d", ""))]
    assert len(values) == 4
    return np.asarray(values).reshape(2, 2)


def _alkene_side_product(tree):
    groups = {number: [path for path in bond_paths(tree) if f"bond-{number}" in path.get("class", "").split()] for number in (0, 1, 2)}
    axis = _line_endpoints(groups[1][0])
    origin, direction = axis[0], axis[1] - axis[0]
    midpoint = axis.mean(axis=0)
    sides = []
    for number in (0, 2):
        points = np.concatenate([_line_endpoints(path) for path in groups[number]])
        endpoint = points[np.argmax(np.linalg.norm(points - midpoint, axis=1))]
        offset = endpoint - origin
        sides.append(direction[0] * offset[1] - direction[1] * offset[0])
    return sides[0] * sides[1]


def test_e_z_alkene_substituents_remain_on_opposite_or_same_vector_sides():
    assert _alkene_side_product(drawing("F/C=C/F")) < 0
    assert _alkene_side_product(drawing("F/C=C\\F")) > 0


def test_disconnected_sodium_and_chloride_ions_and_negative_charge_are_visible():
    tree = drawing("[Na+].[Cl-]")
    assert not bond_paths(tree)
    assert all(any(path.get("class") == f"atom-{index}" for path in paths(tree)) for index in (0, 1))
    minus_rectangles = []
    for path in paths(tree):
        if path.get("class") != "atom-1" or "Q" in path.get("d", ""):
            continue
        numbers = np.asarray([float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path.get("d", ""))]).reshape(-1, 2)
        extent = np.ptp(numbers, axis=0)
        if extent[1] > 0 and extent[0] / extent[1] > 3:
            minus_rectangles.append(path)
    assert minus_rectangles  # horizontal minus glyph, not just atom labels
    description = tree.find(f"{{{SVG_NAMESPACE}}}desc").text
    assert "[Na+]" in description and "[Cl-]" in description


def test_requested_highlights_use_passed_atom_indices_without_canonical_reordering():
    mol = Chem.MolFromSmiles("OCCN")  # canonical NCCO has the opposite terminal order
    assert Chem.MolToSmiles(mol) == "NCCO"
    tree = ET.fromstring(render_molecule_svg(mol, highlight_atoms=[0, 3, 0]))
    ellipses = list(tree.iter(f"{{{SVG_NAMESPACE}}}ellipse"))
    assert {ellipse.get("class") for ellipse in ellipses} == {"atom-0", "atom-3"}
    assert mol.GetAtomWithIdx(0).GetSymbol() == "O"
    assert mol.GetAtomWithIdx(3).GetSymbol() == "N"


def test_drawing_does_not_mutate_input_graph_stereo_indices_or_3d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))
    assert AllChem.EmbedMolecule(mol, randomSeed=1729) == 0
    before_binary = mol.ToBinary()
    before_positions = mol.GetConformer().GetPositions().copy()
    before_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    assert render_molecule_svg(mol, highlight_atoms=[1])
    assert mol.ToBinary() == before_binary
    assert Chem.MolToSmiles(mol, isomericSmiles=True) == before_smiles
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), before_positions)
    assert mol.GetConformer().Is3D()


def test_svg_is_responsive_with_unique_meaningful_accessibility_references():
    left, right = drawing("CCO", width=800, height=300), drawing("CCO", width=800, height=300)
    assert left.get("width") == left.get("height") == "100%"
    assert left.get("viewBox") == "0 0 800 300"
    assert left.get("role") == "img"
    title, description = left.find(f"{{{SVG_NAMESPACE}}}title"), left.find(f"{{{SVG_NAMESPACE}}}desc")
    assert title.get("id") == left.get("aria-labelledby")
    assert description.get("id") == left.get("aria-describedby")
    assert "C2H6O" in title.text
    assert "Isomeric SMILES: CCO" in description.text
    assert left.get("aria-labelledby") != right.get("aria-labelledby")


@pytest.mark.parametrize("mol,options", [(None, {}), ("CC", {}), (Chem.MolFromSmiles(""), {}),
    (Chem.MolFromSmiles("CC"), {"highlight_atoms": [-1]}),
    (Chem.MolFromSmiles("CC"), {"highlight_atoms": [2]}),
    (Chem.MolFromSmiles("CC"), {"highlight_atoms": [True]}),
    (Chem.MolFromSmiles("CC"), {"highlight_atoms": [np.bool_(False)]}),
    (Chem.MolFromSmiles("CC"), {"highlight_atoms": [1.5]}),
    (Chem.MolFromSmiles("CC"), {"width": 0}),
    (Chem.MolFromSmiles("CC"), {"height": -10}),
    (Chem.MolFromSmiles("CC"), {"width": True}),
    (Chem.MolFromSmiles("CC"), {"width": np.bool_(True)})])
def test_invalid_inputs_return_no_depiction_instead_of_an_inaccurate_fallback(mol, options):
    assert render_molecule_svg(mol, **options) is None


def test_failed_rdkit_preparation_returns_none_without_fake_svg():
    with patch.object(rdMolDraw2D, "PrepareAndDrawMolecule", side_effect=RuntimeError("drawing unavailable")):
        assert render_molecule_svg(Chem.MolFromSmiles("C#C")) is None


def test_unavailable_rdkit_drawing_module_returns_none_without_fake_svg():
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "rdkit.Chem.Draw":
            raise ImportError("drawing support unavailable")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=unavailable):
        assert render_molecule_svg(Chem.MolFromSmiles("CC")) is None
