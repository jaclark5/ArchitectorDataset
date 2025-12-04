
from architector import build_complex

inpDict = {
    'core': {
        'metal': 'Fe',
        'coreCN': 2,
        'coreTypes': ['bent_120'],
        'smiles': '[Fe]'
    },
    'ligands': [
        {
            'smiles': 'N([H])[H]',
            'coordList': [0],
            'ligType': 'mono',
            'bondType': 'X',
            'charge': 0
        },
        {
            'smiles': 'P([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[H]',
            'coordList': [0],
            'ligType': 'mono',
            'bondType': 'L',
            'charge': 0
        }
    ],
    'parameters': {
        'metal_ox': 2,
        'assemble_method': 'GFN-FF',
        'n_symmetries': 5,
        'n_conformers': 1,
        'return_only_1': True,
        'is_actinide': False,
        'original_metal': 'Fe',
        'force_trans_oxos': False
    }
}

out = build_complex(inpDict)

if out:
    print("\nGot it")
else:
    print("\nDon't got it")

print(out)