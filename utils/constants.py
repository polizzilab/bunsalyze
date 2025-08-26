from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF = 3.3
ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF = 4.0
S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF = 4.3
MIN_HBOND_ANGLE = 110
MIN_HBOND_DISTANCE = 1.5
H_TO_H_CLASH_DIST = 1.5


@dataclass
class DonorHydrogen:
    name: str
    coord: np.ndarray
    is_engaged: bool = False
    engaged_to: Optional['PolarAtom'] = None

    def engage(self, other):
        self.is_engaged = True
        self.engaged_to = other


@dataclass
class PolarAtom:
    name: str
    coord: np.ndarray
    donor_count: int
    acceptor_count: int
    parent_group_identifier: tuple
    element: str
    donor_hydrogens: list[DonorHydrogen]
    max_donor_count: int = field(init=False)
    max_acceptor_count: int = field(init=False)
    is_buried: Optional[bool] = None

    def __post_init__(self):
        self.max_donor_count = self.donor_count
        self.max_acceptor_count = self.acceptor_count


aa_short_to_long = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET', 'X': 'XAA'}
aa_long_to_short = {x: y for y, x in aa_short_to_long.items()}

aa_to_sc_hbond_donor_to_heavy_atom = {
    'C': {'SG': [('HG',)]},
    'H': {
        'ND1': [('HD1',)], 
        'NE2': [('HE2',)]
    }, 
    'K': {'NZ': [('HZ1', '1HZ'), ('HZ2', '2HZ'), ('HZ3', '3HZ')]},
    'N': {'ND2': [('HD21', '1HD2'), ('HD22', '2HD2')]},
    'Q': {'NE2': [('HE21', '1HE2'), ('HE22', '2HE2')]},
    'R': {
        'NE': [('HE',)],
        'NH1': [('HH11', '1HH1'), ('HH12', '2HH1')],
        'NH2': [('HH21', '1HH2'), ('HH22', '2HH2')],
    },
    'S': {'OG': [('HG',)]},
    'T': {'OG1': [('HG1',)]},
    'W': {'NE1': [('HE1',)]},
    'Y': {'OH': [('HH',)]},
}

# Add backbone N-H for all amino acids
for aa in aa_short_to_long:
    if aa == 'X':
        continue
    subdict = aa_to_sc_hbond_donor_to_heavy_atom.setdefault(aa, {})
    subdict['N'] = [('H',), ('1H', 'H1'), ('2H', 'H2'), ('3H', 'H3')]

aa_to_sc_hbond_acceptor_heavy_atom = {
    'G': ['O'],
    'A': ['O'],
    'S': ['O', 'OG'],
    'C': ['O', 'SG'],
    'T': ['O', 'OG1'],
    'P': ['O'],
    'V': ['O'],
    'M': ['O', 'SD'],
    'N': ['O', 'OD1'],
    'I': ['O'],
    'L': ['O'],
    'D': ['O', 'OD1', 'OD2'],
    'E': ['O', 'OE1', 'OE2'],
    'K': ['O'],
    'Q': ['O', 'OE1'],
    'H': ['O'],
    'F': ['O'],
    'R': ['O'],
    'Y': ['O', 'OH'],
    'W': ['O']
}

# Add backbone OXT to all amino acids:
for aa, sublist in aa_to_sc_hbond_acceptor_heavy_atom.items():
    sublist.append('OXT')

# Create a mapping of amino acids to all polar atoms that can act as either donors, acceptors, or both.
aa_to_polar_atoms = defaultdict(set)
for aa, subdict in aa_to_sc_hbond_donor_to_heavy_atom.items():
    for key in subdict:
        aa_to_polar_atoms[aa].add(key)
for aa, sublist in aa_to_sc_hbond_acceptor_heavy_atom.items():
    for key in sublist:
        aa_to_polar_atoms[aa].add(key)