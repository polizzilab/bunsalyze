import io
import math
import prody as pr
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem

from .constants import PolarAtom, DonorHydrogen


tbl = Chem.GetPeriodicTable()


def num_lone_pairs(atom) -> int:
    v = tbl.GetNOuterElecs(atom.GetAtomicNum())
    c = atom.GetFormalCharge()
    b = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
    return max(0, math.floor(0.5 * (v - c - b)))


def compute_ligand_capacity(rdmol: Chem.Mol) -> dict:
    """
    For each N or O atom:
      - Donor capacity: total H count (implicit + explicit)
      - Acceptor capacity: inferred from hybridization, bonds, and delocalization
    """
    capacity = {}
    for atom in rdmol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in (7, 8):
            continue

        idx = atom.GetIdx()
        resinfo = atom.GetPDBResidueInfo()
        atom_name = resinfo.GetName().strip() if resinfo else f"{atom.GetSymbol()}{idx}"

        # Get total donor hydrogens attached
        nH = atom.GetTotalNumHs()
        donor = nH
        acceptor = 0

        # Oxygen: acceptor if 2 or fewer bonds and no positive charge.
        if atomic_num == 8:
            if atom.GetTotalValence() <= 2 and atom.GetFormalCharge() <= 0:
                acceptor = num_lone_pairs(atom)

        # Nitrogen: donor with H, some constraints for acceptor
        elif atomic_num == 7:
            degree = atom.GetTotalDegree()
            hyb = atom.GetHybridization()
            is_aromatic = atom.GetIsAromatic()
            charge = atom.GetFormalCharge()

            if (degree < 4) and (charge == 0):
                # Handle sp2 nitrogens, can only accept if less than 3 covalent bonds.
                if (hyb == Chem.rdchem.HybridizationType.SP2) and degree < 3:
                    acceptor = num_lone_pairs(atom)

                # Handle sp3 nitrogens, can only accept if 3 bonds and uncharged.
                if (hyb == Chem.rdchem.HybridizationType.SP3) and degree == 3:
                    acceptor = num_lone_pairs(atom)

        capacity[atom_name] = {'donor': donor, 'acceptor': acceptor}
    return capacity


def get_ligand_polar_atoms(lig_cap: dict, lig_ag: pr.AtomGroup, covalent_hydrogen_max_distance: float = 1.2) -> List[PolarAtom]:
    polar_atoms = []
    for atom, don_acc in lig_cap.items():
        # Skip if not donor or acceptor
        donor_count, acceptor_count = don_acc['donor'], don_acc['acceptor']
        if donor_count == 0 and acceptor_count == 0:
            continue

        # If donor, get the associated hydrogens.
        donor_hydrogens = []
        if donor_count > 0:
            covalent_hydrogens = lig_ag.select(f'element H within {covalent_hydrogen_max_distance} of (name {atom})')
            donor_hs = covalent_hydrogens.getNames()
            donor_hs_coords = covalent_hydrogens.getCoords()
            donor_hydrogens = [DonorHydrogen(name=x[0], coord=x[1]) for x in zip(donor_hs, donor_hs_coords)]


        # Get a unique identifier for the residue the atom belongs to and its coordinates.
        atom_ag = lig_ag.select(f'name {atom}')

        parent_group_id = []
        parent_group_id.append(atom_ag.getChids()[0])
        parent_group_id.append(atom_ag.getResnames()[0])
        parent_group_id.append(int(atom_ag.getResnums()[0]))
        parent_group_id.append(atom_ag.getIcodes()[0])
        parent_group_id = tuple(parent_group_id)
        atom_coord = atom_ag.getCoords()[0]

        polar_atoms.append(PolarAtom(
            name=atom,
            coord=atom_coord,
            donor_count=donor_count,
            acceptor_count=acceptor_count,
            donor_hydrogens=donor_hydrogens,
            parent_group_identifier=parent_group_id,
            element=atom_ag.getElements()[0]
        ))
    return polar_atoms