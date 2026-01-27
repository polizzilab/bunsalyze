import prody as pr
import numpy as np
from copy import deepcopy
from typing import List
from .constants import PolarAtom, DonorHydrogen, aa_to_polar_atoms, aa_to_sc_hbond_donor_to_heavy_atom, aa_to_sc_hbond_acceptor_heavy_atom, aa_long_to_short, DEFAULT_NCAA_DICT


def flat_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def get_protein_polar_atoms(protein_ag: pr.AtomGroup, ncaa_dict: dict, use_sulfur_acceptors: bool = True, use_ca_donors: bool = True) -> List[PolarAtom]:

    # By default, use the DEFAULT_NCAA_DICT and update it with any user-provided ncaa_dict entries.
    ncaa_dict_ = deepcopy(DEFAULT_NCAA_DICT)
    ncaa_dict_.update(deepcopy(ncaa_dict))

    polar_atoms = []
    for residue in protein_ag.iterResidues():

        parent_group_id = (
            residue.getChids()[0],
            residue.getResnames()[0],
            int(residue.getResnums()[0]),
            residue.getIcodes()[0]
        )

        # Handles non-canonical amino acids 
        if residue.getResname() in ncaa_dict_:
            for atom, (acceptor_count, donor_list) in ncaa_dict_[residue.getResname()].items():
                atom_sele = residue.select(f'name {atom}')
                if atom_sele is None:
                    raise ValueError(f'Atom {atom} not found in residue {residue.getResname()} {residue.getResnum()} {residue.getIcode()} for ncaa_dict entry. If any atoms are not present or are renamed, please update the ncaa_dict passed into main function.')

                coord = atom_sele.getCoords()[0]
                
                donor_hydrogens = []
                for atom in donor_list:
                    h_coord_sele = residue.select(f'name {atom}')
                    if h_coord_sele is None:
                        raise ValueError(f'Donor hydrogen {atom} not found in residue {residue.getResname()} {residue.getResnum()} {residue.getIcode()} for ncaa_dict entry. If any atoms are not present or are renamed, please update the ncaa_dict passed into main function.')
                    h_coord = h_coord_sele.getCoords()[0]
                    donor_hydrogens.append(DonorHydrogen(name=atom, coord=h_coord))
                
                polar_atoms.append(PolarAtom(
                    name=atom,
                    coord=coord,
                    donor_count=len(donor_hydrogens),
                    acceptor_count=acceptor_count,
                    donor_hydrogens=donor_hydrogens,
                    parent_group_identifier=parent_group_id,
                    element=atom_sele.getElements()[0]  # Infer element from name as first alphabetic character.
                ))

            # Skip to next residue after processing ncaa.
            continue

        # Skip non-standard amino acids not in the provided ncaa_dict.
        if residue.getResname() not in aa_long_to_short:
            raise ValueError(f'Residue {residue.getResname()} not in standard amino acids, known NCAAs, and no ncaa_dict provided.')

        # Get the residue name and atom data.
        aa_short = aa_long_to_short[residue.getResname()]
        curr_atom_names = residue.getNames()
        curr_atom_coords = residue.getCoords()

        # Identify hbond-capable polar atoms in the residue.
        polar_atoms_set = aa_to_polar_atoms[aa_short]
        polar_atoms_mask = np.array([x in polar_atoms_set for x in curr_atom_names])

        # Create PolarAtom objects for all polar atoms in the residue.
        for polar_atom, polar_atom_coord in zip(curr_atom_names[polar_atoms_mask], curr_atom_coords[polar_atoms_mask]):

            if not use_sulfur_acceptors:
                print('Skipping sulfur atom:', residue.getResname(), residue.getResnum(), residue.getIcode(), polar_atom)
                continue

            if not use_ca_donors:
                if polar_atom == 'CA':
                    print('Skipping CA atom:', residue.getResname(), residue.getResnum(), residue.getIcode(), polar_atom)
                    continue

            # Identify any donor hydrogens associated with the polar atom.
            donor_hydrogens = []
            if polar_atom in aa_to_sc_hbond_donor_to_heavy_atom[aa_short]:
                donor_hydrogen_mask = np.array([x in flat_list(aa_to_sc_hbond_donor_to_heavy_atom[aa_short][polar_atom]) for x in curr_atom_names])

                num_donor_hs = np.sum(donor_hydrogen_mask)
                if (
                    (not len(aa_to_sc_hbond_donor_to_heavy_atom[aa_short][polar_atom]) == num_donor_hs) and
                    (not (polar_atom == 'N' and num_donor_hs in (1, 3))) and
                    (not (aa_short == 'P' and polar_atom == 'N' and num_donor_hs in (0, 2))) and
                    (not (aa_short == 'H' and polar_atom in ('ND1', 'NE2') and num_donor_hs == 0)) and 
                    (not (aa_short == 'C' and polar_atom == 'SG' and num_donor_hs == 0)) and
                    (not (aa_short == 'G' and polar_atom == 'CA' and num_donor_hs == 2))
                ):
                    print(f"Warning: {residue.getResname()} {residue.getResnum()} {residue.getIcode()} {polar_atom} has {num_donor_hs} donor hydrogens (names: {curr_atom_names}), but expected {len(aa_to_sc_hbond_donor_to_heavy_atom[aa_short][polar_atom])}.")

                for donor_hydrogen, donor_hydrogen_coord in zip(curr_atom_names[donor_hydrogen_mask], curr_atom_coords[donor_hydrogen_mask]):
                    donor_hydrogens.append(
                        DonorHydrogen(name=donor_hydrogen, coord=donor_hydrogen_coord)
                    )

            # Identify the number of acceptor lone pairs associated with the polar atom.
            acceptor_count = 0
            if polar_atom in aa_to_sc_hbond_acceptor_heavy_atom[aa_short]:

                if polar_atom in aa_to_sc_hbond_acceptor_heavy_atom[aa_short]:
                    acceptor_count = 2
                
                if aa_short == 'M':
                    acceptor_count = 1
                
                # Cysteine wouldn't accept in disulfide form, 
                # so check that it has a donor hydrogen since we will not generate designs with thiolates (-S⁻)
                if aa_short == 'C': 
                    if len(donor_hydrogens) == 0:
                        continue
                    acceptor_count = 1

                # The only exception to acceptor_count == 2 is for histidine nitrogens where
                if (aa_short == 'H') and (polar_atom in ('ND1', 'NE2')) and (len(donor_hydrogens) == 0):
                    acceptor_count = 1

            # Infer element from name as first alphabetic character.
            element = [x for x in polar_atom if x.isalpha()][0]

            polar_atoms.append(PolarAtom(
                name=polar_atom,
                coord=polar_atom_coord,
                donor_count=len(donor_hydrogens),
                acceptor_count=acceptor_count,
                donor_hydrogens=donor_hydrogens,
                parent_group_identifier=parent_group_id,
                element=element
            ))

    return polar_atoms