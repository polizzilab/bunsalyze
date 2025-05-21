import os
import io
import json
from pathlib import Path
from pprint import pprint
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import freesasa
import prody as pr
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

from bbunsalyze.utils.calc_ligand_dons_accs import get_ligand_polar_atoms, compute_ligand_capacity
from bbunsalyze.utils.calc_protein_dons_accs import get_protein_polar_atoms
from bbunsalyze.utils.burial_calc import compute_fast_ligand_burial_mask
from bbunsalyze.utils.graph import PolarAtomGraph

freesasa.setVerbosity(1)
RDLogger.DisableLog('rdApp.*') # Disables all RDKit logging


def parse_complex_and_build_rdkit_ligand(pr_complex, smiles):
    prot_ag = pr_complex.select('protein').copy()
    lig_ag = pr_complex.select('not protein').copy()

    if len(set(lig_ag.getResindices())) > 1:
        raise NotImplementedError(
            'Multiple residues in ligand.'
            'Only single residue ligands are currently supported since we only take one smiles string as input.'
        )

    smi_mol = Chem.MolFromSmiles(smiles)

    buff = io.StringIO()
    pr.writePDBStream(buff, lig_ag.copy())
    lig_mol = Chem.MolFromPDBBlock(buff.getvalue())
    lig_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, lig_mol)

    return prot_ag, lig_ag, smi_mol, lig_mol


def set_burial_annotations(ligand_polar_atoms, protein_polar_atoms, ca_coords, input_path, sasa_threshold, silent):
    ligand_burial_mask = compute_fast_ligand_burial_mask(ca_coords, np.array([x.coord for x in ligand_polar_atoms]))
    protein_burial_mask = compute_fast_ligand_burial_mask(ca_coords, np.array([x.coord for x in protein_polar_atoms]))

    freesasa_struct = freesasa.Structure(str(input_path), options={'hetatm': True, 'hydrogen': False})
    sasa_data = freesasa.calc(freesasa_struct)

    ligand_burial_annotations = {'ligand_atoms_in_hull': [], 'ligand_atoms_buried_sasa': [], 'ligand_atoms_sasa': {}}
    if not silent: print('Ligand Atom SASA: (< 1.0 A^2 == buried)')
    for i, atom in enumerate(ligand_polar_atoms):
        atom_name = atom.name
        chain, resname, resnum, *_ = atom.parent_group_identifier
        sasa = freesasa.selectArea([f's1, name {atom_name} and resn {resname} and chain {chain} and resi {resnum}'], freesasa_struct, sasa_data)['s1']

        ligand_burial_annotations['ligand_atoms_sasa'][atom_name] = sasa
        if ligand_burial_mask[i].item():
            ligand_burial_annotations['ligand_atoms_in_hull'].append(atom_name)
        if sasa < sasa_threshold:
            ligand_burial_annotations['ligand_atoms_buried_sasa'].append(atom_name)

        atom.is_buried = ligand_burial_mask[i].item() and (sasa < sasa_threshold)
        if not silent: print(f'\t{atom_name} {sasa}')

    for i, atom in enumerate(protein_polar_atoms):
        atom_name = atom.name
        chain, resname, resnum, *_ = atom.parent_group_identifier
        sasa = freesasa.selectArea([f's1, name {atom_name} and resn {resname} and chain {chain} and resi {resnum}'], freesasa_struct, sasa_data)['s1']
        atom.is_buried = protein_burial_mask[i].item() and (sasa < sasa_threshold)
    
    return ligand_burial_annotations


def compute_capacity_score(ligand_polar_atoms, protein_polar_atoms) -> dict:
    output_data = []
    for idx, polar_atoms in enumerate((ligand_polar_atoms, protein_polar_atoms)):
        residue_to_buried_atoms = defaultdict(list)
        residue_fraction_unsatisfied = {}
        residue_fraction_buried_unsatisfied = {}

        # Map from residue to polar atoms
        for atom in polar_atoms:
            residue_to_buried_atoms[atom.parent_group_identifier].append(atom)
        
        # Loop over each residue and compute the fraction of capacity not satisfied
        for residue, atoms in residue_to_buried_atoms.items():
            remaining_capacity, max_capacity, buried_remaining_capacity, max_buried_capacity = 0, 0, 0, 0
            for atom in atoms:
                if (atom.name not in ('N', 'O', 'OXT', 'SD')) or (idx == 0):
                    remaining_capacity += atom.donor_count + atom.acceptor_count
                    max_capacity += atom.max_donor_count + atom.max_acceptor_count
                    if atom.is_buried:
                        buried_remaining_capacity += atom.donor_count + atom.acceptor_count
                        max_buried_capacity += atom.max_donor_count + atom.max_acceptor_count

            # Calculate the fraction of capacity satisfied for buried and non-buried atoms.
            fraction_buried_capacity_satisfied = (max_buried_capacity - buried_remaining_capacity) / max_buried_capacity if max_buried_capacity else 1.0
            fraction_capacity_satisfied = (max_capacity - remaining_capacity) / max_capacity if max_capacity else 1.0

            # If not fully satisfied, add to the unsatisfied lists.
            if fraction_buried_capacity_satisfied < 1.0:
                residue_fraction_unsatisfied[residue] = 1 - fraction_capacity_satisfied
                residue_fraction_buried_unsatisfied[residue] = 1 - fraction_buried_capacity_satisfied
        
        output_data.append((residue_fraction_unsatisfied, residue_fraction_buried_unsatisfied))
    
    return {
        'ligand_buried_fraction_unsat': output_data[0][1], 
        'protein_buried_fraction_unsat': output_data[1][1],
        'ligand_fraction_unsat': output_data[0][0], 
        'protein_fraction_unsat': output_data[1][0]
    }



def main(input_path: os.PathLike, protein_complex: pr.AtomGroup, smiles: str, sasa_threshold: float = 1.0, silent: bool = True):

    # Load the relevant protein and ligand data.
    prot_ag, lig_ag, smi_mol, lig_mol = parse_complex_and_build_rdkit_ligand(protein_complex, smiles)
    lig_cap = compute_ligand_capacity(lig_mol)
    ca_coords = prot_ag.select('name CA').getCoords()

    # Get the hbond-able polar atoms.
    ligand_polar_atoms = get_ligand_polar_atoms(lig_cap, lig_ag)
    protein_polar_atoms = get_protein_polar_atoms(prot_ag)
    ligand_burial_annotations = set_burial_annotations(ligand_polar_atoms, protein_polar_atoms, ca_coords, input_path, sasa_threshold=sasa_threshold, silent=silent)

    # Build a radius graph of the polar atoms and compute the buns for the ligand and protein.
    g = PolarAtomGraph(ligand_polar_atoms, protein_polar_atoms, run_hydrogen_atom_clash_check=True)
    ligand_buns = g.compute_ligand_buns()
    protein_buns = g.compute_protein_buns()

    fraction_unsat_dicts = compute_capacity_score(ligand_polar_atoms, protein_polar_atoms)

    output = {
        'input_path': str(input_path), 
        'ligand_buns': [(i.name, *i.parent_group_identifier) for i in ligand_buns], 
        'protein_buns': [(i.name, *i.parent_group_identifier) for i in protein_buns], 
        **fraction_unsat_dicts,
        **ligand_burial_annotations,
    }
    
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Analyze protein-ligand complexes for buried unsatisfied polar atoms (BUNs).")
    parser.add_argument("input_path", type=str, help="Path to the PDB file containing the protein-ligand complex")
    parser.add_argument("smiles", type=str, help="SMILES string of the ligand")
    parser.add_argument("--sasa_threshold", type=float, default=1.0, help="SASA threshold for burial (default: 1.0 Å²)")
    parser.add_argument("--output", type=str, help="Output file path (default: print to stdout)")
    args = parser.parse_args()

    # input_path = 'test.pdb'
    # smiles = r"O=C(C1=C([H])N2C([H])=C(N([H])C(C3=C([H])C([H])=C(O[H])C([H])=C3[H])=O)C([H])=C([H])C2=N1)N4C([H])([H])[C@](C([H])([H])Cl)([H])C5=C4C([H])=C(O[H])C6=C([H])C([H])=C([H])C(C([H])([H])[H])=C65"
    # input_path = 'epic_1.pdb'
    # smiles = r"CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O"
    # input_path = '000_chunk_0_seq_39_model_0_rosetta_emin.pdb'
    # smiles = r"O=C(C([H])(C([H])(C([H])(/C([H])=C1O[C@@]([H])(C([H])([C@]([H])([C@@]2(/C([H])=C([C@]([H])(C([H])(C([H])(C([H])(C([H])(C([H])([H])[H])[H])[H])[H])[H])O[H])\\[H])[H])O[H])[H])[C@]2([H])C\\1([H])[H])[H])[H])[H])[O-]"
    # input_path = './exa05.pdb'
    # smiles = r"CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O"
    # input_path = './1bs2_1.pdb'
    # smiles = r"O=C([O-])C([NH3+])CCCNC(N)=[NH2+]"
    # input_path = './3lvp_1.pdb'
    # smiles = r'Cn1cc(c2c1cc(c(n2)OC)OC)c3cc4c(ccnc4[nH]3)Cl'
    # input_path = './1th6_1.pdb'
    # smiles = r'CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3'

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {args.input_path} does not exist.")

    complex_ = pr.parsePDB(str(args.input_path))
    results = main(args.input_path, complex_, args.smiles, sasa_threshold=args.sasa_threshold, silent=False)
    
    if args.output:
        pprint(results)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        pprint(results)
