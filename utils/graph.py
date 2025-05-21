from typing import Sequence
from dataclasses import dataclass

import torch
torch.set_num_threads(2)
import numpy as np
import networkx as nx

from .constants import (
    PolarAtom, DonorHydrogen, ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF, 
    ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF, S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF, 
    MIN_HBOND_ANGLE, MIN_HBOND_DISTANCE, H_TO_H_CLASH_DIST
)


class PolarAtomGraph:
    def __init__(
        self, ligand_polar_atoms: Sequence[PolarAtom], protein_polar_atoms: Sequence[PolarAtom], run_hydrogen_atom_clash_check: bool
    ):

        self.hbond_max_distance = S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF
        self.ligand_polar_atoms = ligand_polar_atoms
        self.protein_polar_atoms = protein_polar_atoms
        self.run_hydrogen_atom_clash_check = run_hydrogen_atom_clash_check

        # Compute a list of all polar atom coordinates and masks tracking which are from the ligand.
        lig_coords, prot_coords = self._get_ligand_coords(), self._get_protein_coords()
        self.all_coords = torch.cat([lig_coords, prot_coords])
        self.is_lig_mask = torch.zeros(self.all_coords.shape[0], dtype=torch.bool)
        self.is_lig_mask[:len(lig_coords)] = True

        self.ligand_atom_indices = np.arange(len(ligand_polar_atoms))
        self.protein_atom_indices = np.arange(len(protein_polar_atoms)) + len(ligand_polar_atoms)
    
        # Compute a graph defined by atoms within the cutoff distance.
        self.distance_matrix = torch.cdist(self.all_coords, self.all_coords, p=2)
        masked_matrix = self.distance_matrix < self.hbond_max_distance
        masked_matrix = masked_matrix & (~torch.eye(masked_matrix.shape[0], dtype=torch.bool))
        self.edge_index = masked_matrix.nonzero(as_tuple=False)

        self.graph = nx.from_edgelist(self.edge_index.numpy())
        self.graph.add_nodes_from(np.arange(self.all_coords.shape[0]))

    def _get_ligand_coords(self):
        return torch.from_numpy(np.array([atom.coord for atom in self.ligand_polar_atoms]))

    def _get_protein_coords(self):
        return torch.from_numpy(np.array([atom.coord for atom in self.protein_polar_atoms]))

    def _get_neighborhood(self, atom_index):
        atom_neighbors = list(self.graph.neighbors(atom_index))

        neighbor_ligand_atom_indices = sorted([
            (x, self.distance_matrix[atom_index, x]) for x in atom_neighbors if x in self.ligand_atom_indices
        ], key=lambda x: x[1])
        neighbor_protein_atom_indices = sorted([
            (x - len(self.ligand_polar_atoms), self.distance_matrix[atom_index, x]) for x in atom_neighbors if x not in self.ligand_atom_indices
        ], key=lambda x: x[1])

        return [x[0] for x in neighbor_ligand_atom_indices], [x[0] for x in neighbor_protein_atom_indices]
    
    def _consume_neighborhood(self, curr_polar_atom, neighbor_polar_atoms):
        found_valid_hbond = False
        for neighbor_polar_atom in neighbor_polar_atoms:
            # Check the current atom and neighbor atom for donor hydrogens.
            for hydrogen in curr_polar_atom.donor_hydrogens:
                found_valid_hbond = (
                    found_valid_hbond or 
                    is_valid_hbond(curr_polar_atom, hydrogen, neighbor_polar_atom, hydrogen_clash_check=self.run_hydrogen_atom_clash_check)
                )
            for neighbor_hydrogen in neighbor_polar_atom.donor_hydrogens:
                found_valid_hbond = (
                    found_valid_hbond or 
                    is_valid_hbond(neighbor_polar_atom, neighbor_hydrogen, curr_polar_atom, hydrogen_clash_check=self.run_hydrogen_atom_clash_check)
                )
        return found_valid_hbond
    
    def compute_ligand_buns(self):
        unsatisfied_ligand_atoms = []
        for ligand_atom_index in self.ligand_atom_indices:
            neighbor_ligand_atom_indices, neighbor_protein_atom_indices = self._get_neighborhood(ligand_atom_index)
            curr_polar_atom = self.ligand_polar_atoms[ligand_atom_index]
            assert curr_polar_atom.is_buried is not None, f'Need to set atom burial.'

            neighbor_polar_atoms = (
                [self.protein_polar_atoms[x] for x in neighbor_protein_atom_indices] + 
                [self.ligand_polar_atoms[x] for x in neighbor_ligand_atom_indices]
            )

            found_valid_hbond = self._consume_neighborhood(curr_polar_atom, neighbor_polar_atoms)
            if (not found_valid_hbond) and curr_polar_atom.is_buried:
                unsatisfied_ligand_atoms.append(curr_polar_atom)

        return unsatisfied_ligand_atoms
    
    def compute_protein_buns(self):
        unsatisfied_protein_atoms = []
        for protein_atom_index in self.protein_atom_indices:
            neighbor_ligand_atom_indices, neighbor_protein_atom_indices = self._get_neighborhood(protein_atom_index)
            curr_polar_atom = self.protein_polar_atoms[protein_atom_index - len(self.ligand_polar_atoms)]
            assert curr_polar_atom.is_buried is not None, f'Need to set atom burial.'

            neighbor_polar_atoms = (
                [self.ligand_polar_atoms[x] for x in neighbor_ligand_atom_indices] +
                [self.protein_polar_atoms[x] for x in neighbor_protein_atom_indices]
            )

            found_valid_hbond = self._consume_neighborhood(curr_polar_atom, neighbor_polar_atoms)

            if (
                (not found_valid_hbond) and 
                (not curr_polar_atom.name in ('N', 'O', 'OXT')) and 
                curr_polar_atom.is_buried and
                (not (curr_polar_atom.element == 'S' and curr_polar_atom.parent_group_identifier[1] == 'MET')) # Met sulfur atoms may accept but should not be counted as buns.
            ):
                unsatisfied_protein_atoms.append(curr_polar_atom)

        return unsatisfied_protein_atoms


def is_valid_hbond(
    donor_atom: PolarAtom, donor_hydrogen: DonorHydrogen, acceptor: PolarAtom, 
    hydrogen_clash_check: bool, debug: bool = False
):
    # Check that donor and acceptor are possible.

    threshold_distance = None
    if (donor_atom.element in ('N', 'O') and acceptor.element in ('N', 'O')):
        threshold_distance = ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF
    elif (
        (donor_atom.element == 'S' and acceptor.element in ('N', 'O')) or
        (donor_atom.element in ('N', 'O') and acceptor.element == 'S')
    ):
        threshold_distance = ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF
    elif (donor_atom.element == 'S' and acceptor.element == 'S'):
        threshold_distance = S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF
    else:
        raise NotImplementedError(f'Unsupported donor-acceptor pair: {donor_atom.element}, {acceptor.element}')

    # Check that donor hydrogen is not already doing something else.
    if donor_hydrogen.is_engaged:
         # It's possible that this hbond has already been detected in the other direction so check for that.
        if donor_hydrogen.engaged_to is acceptor:
            if debug: print(donor_atom.name, donor_hydrogen.name, acceptor.name, True)
            return True

        # If the donor hydrogen is already engaged to something else, we can't use it for this hbond.
        # if debug: print(donor_atom.name, donor_hydrogen.name, acceptor.name, False, 'donor hydrogen already engaged')
        return False
    
    # Check if the donor and acceptor have capacity for hbonding.
    if not ((donor_atom.donor_count > 0) and (acceptor.acceptor_count > 0)):
        if debug: print(donor_atom.name, donor_hydrogen.name, acceptor.name, False, 'not enough donors or acceptors')
        return False

    # Check if the donor-acceptor distance is within the cutoff
    distance = np.linalg.norm(donor_hydrogen.coord - acceptor.coord)
    if (distance > threshold_distance) or (distance < MIN_HBOND_DISTANCE):
        if debug: print(donor_atom.name, donor_hydrogen.name, acceptor.name, False, f'distance too far {distance}')
        return False

    # Angle between D->H and A->H vectors should be greater than MIN_HBOND_ANGLE
    angle = np.rad2deg(np.arccos((
        np.dot(donor_hydrogen.coord - donor_atom.coord, donor_hydrogen.coord - acceptor.coord) /
        (np.linalg.norm(donor_hydrogen.coord - donor_atom.coord) * np.linalg.norm(donor_hydrogen.coord - acceptor.coord))),
    ))
    if angle < MIN_HBOND_ANGLE:
        if debug: print(donor_atom.name, donor_hydrogen.name, acceptor.name, False, f'angle too small {angle}')
        return False

    # TODO: check that donor hydrogen does not clash with any hydrogens on the acceptor?
    if hydrogen_clash_check:
        for acc_donor in acceptor.donor_hydrogens:
            if np.linalg.norm(donor_hydrogen.coord - acc_donor.coord) < H_TO_H_CLASH_DIST:
                if debug: print(donor_atom.name, donor_hydrogen.name, acceptor.name, False, f'clash with {acc_donor.name}')
                return False

    # Adjust the donor and acceptor counts for the hydrogen bond
    donor_atom.donor_count -= 1
    acceptor.acceptor_count -= 1
    donor_hydrogen.engage(acceptor)
    if debug: print(donor_atom.name, donor_hydrogen.name, acceptor.name, True)
    return True


