"""Standard RDKit molecular SVG depiction without altering the submitted graph."""
from __future__ import annotations

import operator
from numbers import Integral
from uuid import uuid4
from xml.etree import ElementTree as ET


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NAMESPACE)


def render_molecule_svg(mol, highlight_atoms=None, width=620, height=420):
    """Return a responsive, accessible RDKit SVG or None when drawing fails.

    The input molecule is copied without canonicalizing or reordering its atoms.
    Highlights therefore use the caller's atom indices. Coordinates and drawing
    preparation are applied only to the copy. There is no hand-drawn fallback.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D

        if mol is None or not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() == 0:
            return None
        if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, Integral) or not isinstance(height, Integral):
            return None
        width, height = operator.index(width), operator.index(height)
        if width <= 0 or height <= 0:
            return None
        indices = set()
        for atom in () if highlight_atoms is None else highlight_atoms:
            if isinstance(atom, bool) or not isinstance(atom, Integral):
                return None
            index = operator.index(atom)
            if not 0 <= index < mol.GetNumAtoms():
                return None
            indices.add(index)

        drawing_mol = Chem.Mol(mol)
        smiles = Chem.MolToSmiles(drawing_mol, isomericSmiles=True)
        formula = rdMolDescriptors.CalcMolFormula(drawing_mol)
        # Force a standard two-dimensional layout, even for a supplied 3D
        # conformer. The graph, atom indices and stereochemical tags are retained.
        drawing_mol.RemoveAllConformers()
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, drawing_mol, highlightAtoms=sorted(indices))
        drawer.FinishDrawing()
        root = ET.fromstring(drawer.GetDrawingText())
        if root.tag != f"{{{SVG_NAMESPACE}}}svg":
            return None

        identifier = uuid4().hex
        title_id, description_id = f"molecule-title-{identifier}", f"molecule-description-{identifier}"
        root.set("width", "100%")
        root.set("height", "100%")
        root.set("viewBox", f"0 0 {width} {height}")
        root.set("preserveAspectRatio", "xMidYMid meet")
        root.set("role", "img")
        root.set("aria-labelledby", title_id)
        root.set("aria-describedby", description_id)
        root.set("focusable", "false")
        title = ET.Element(f"{{{SVG_NAMESPACE}}}title", {"id": title_id})
        title.text = f"Molecular structure: {formula}"
        description = ET.Element(f"{{{SVG_NAMESPACE}}}desc", {"id": description_id})
        description.text = (
            f"RDKit two-dimensional depiction. Isomeric SMILES: {smiles}. "
            f"{mol.GetNumAtoms()} atoms; {mol.GetNumBonds()} bonds. "
            + ("Highlighted submitted atom indices: " + ", ".join(map(str, sorted(indices))) + "." if indices else "No atom highlights.")
        )
        root.insert(0, description)
        root.insert(0, title)
        return ET.tostring(root, encoding="unicode")
    except Exception:
        # Missing drawing support, invalid indices and failed preparation must
        # not produce a simplified graph that misstates chemical structure.
        return None
