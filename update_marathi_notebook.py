"""
Update the Marathi notebook to use train_marathi.py instead of train.py
and fix the configuration paths.
"""

import json

# Load the notebook
with open(r'notebooks\comprehensive_experiments_marathi.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update all cells that reference train.py to use train_marathi.py
updated_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = cell['source']
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        # Replace train.py with train_marathi.py
        if 'src/train.py' in source_str:
            new_source = source_str.replace('src/train.py', 'src/train_marathi.py')
            cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
            updated_count += 1
        
        # Also replace python src/train.py with python src/train_marathi.py
        elif 'python src/train.py' in source_str:
            new_source = source_str.replace('python src/train.py', 'python src/train_marathi.py')
            cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
            updated_count += 1
        
        # Replace !python src/train.py
        elif '!python src/train.py' in source_str:
            new_source = source_str.replace('!python src/train.py', '!python src/train_marathi.py')
            cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
            updated_count += 1

# Save the updated notebook
with open(r'notebooks\comprehensive_experiments_marathi.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Updated {updated_count} cells to use train_marathi.py")
print("✓ Notebook updated successfully!")
