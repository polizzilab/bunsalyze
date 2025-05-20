# BBunsalyze (Ben's Bunsalyze)

### Description

Python module for counting buried polar atoms with unsatisfied hydrogen bonds.

### Installation

Runs with the [LASErMPNN](https://github.com/polizzilab/LASErMPNN) conda environment.

### Usage

```bash
python -m bbunsalyze.bunsalyze ./path/to/pdb_file.pdb 'smiles_string_for_ligand'
```

```text
usage: bunsalyze.py [-h] [--sasa_threshold SASA_THRESHOLD] [--output OUTPUT] input_path smiles

Analyze protein-ligand complexes for buried unsatisfied polar atoms (BUNs).

positional arguments:
  input_path            Path to the PDB file containing the protein-ligand complex
  smiles                SMILES string of the ligand

options:
  -h, --help            show this help message and exit
  --sasa_threshold SASA_THRESHOLD
                        SASA threshold for burial (default: 1.0 Å²). This threshold is combined with alphahull burial. Set to a large value to use only alphahull.
  --output OUTPUT       Output json file path (default: prints to stdout if unset)
```

### TODO:
- [ ] Add additional metrics beyond just binary is/is not bun. Per-residue fraction of capacity satisfied, etc...
- [ ] Add support for multiple ligands