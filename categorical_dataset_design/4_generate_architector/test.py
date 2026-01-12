
from architector import build_complex

ligands=['methyl|N', 'water|N', 'water|N', 'water|N', 'water|N', 'water|N']
inputDict = {
    'core': {
     'metal': 'Fe',
     'coreCN': 6,
     'coreType': ['hexagonal_planar'],
     'smiles': '[Fe]'},
    'ligands': [
        {'smiles': '[C-]([H])([H])[H]', 'coordList': [0], 'ligType': 'mono', 'bondType': 'X'},
        {'smiles': '[O]([H])[H]', 'coordList': [0], 'ligType': 'mono', 'bondType': 'L'},
        {'smiles': '[O]([H])[H]', 'coordList': [0], 'ligType': 'mono', 'bondType': 'L'},
        {'smiles': '[O]([H])[H]', 'coordList': [0], 'ligType': 'mono', 'bondType': 'L'},
        {'smiles': '[O]([H])[H]', 'coordList': [0], 'ligType': 'mono', 'bondType': 'L'},
        {'smiles': '[O]([H])[H]', 'coordList': [0], 'ligType': 'mono', 'bondType': 'L'},
    ],
    'parameters': {
        'debug': True, # NoteHere
        'metal_ox': 0,
        'assemble_method': 'GFN-FF',
        'n_symmetries': 100,
        'n_conformers': 1,
        'return_only_1': True,
        'save_init_geos': True,
        'is_actinide': False,
        'original_metal': 'Fe',
        'force_trans_oxos': False
    }
}

out = build_complex(inputDict)
print(out)