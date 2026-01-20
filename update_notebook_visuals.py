import json
import os

notebook_path = r'c:\Users\Admin\OneDrive\Desktop\olfaction-inspired-ner\notebooks\comprehensive_experiments_marathi.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'analyze_receptor_activations' in source and 'experiment_name=' not in source:
            # First Try: standard formatting
            new_source = source.replace(
                'save_dir=save_dir\n        )',
                'save_dir=save_dir,\n            experiment_name=exp_name\n        )'
            )
            
            # Second Try: if formatting is slightly different (e.g. spaces)
            if new_source == source:
                 new_source = source.replace(
                    'save_dir=save_dir',
                    'save_dir=save_dir, experiment_name=exp_name'
                 )
            
            if new_source != source:
                # Convert back to list of strings, preserving newlines
                cell['source'] = new_source.splitlines(keepends=True)
                updated = True
                print("Found and updated analyze_receptor_activations call.")

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("No matching cell found to update (or it was already updated).")
