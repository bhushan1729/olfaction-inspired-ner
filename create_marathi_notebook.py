"""
Script to generate comprehensive_experiments_marathi.ipynb based on the original notebook
but adapted for the Marathi Naamapadam dataset
"""

import json
import sys

# Load the original notebook
with open(r'notebooks\comprehensive_experiments.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify the title and description cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Update title
        if '# 🧪 Comprehensive Olfaction-Inspired NER Experiments' in source:
            cell['source'] = [
                "# 🧪 Comprehensive Olfaction-Inspired NER Experiments - Marathi Dataset\n",
                "\n",
                "**Goal**: Run 12 different NER experiments on Marathi (ai4bharat/naamapadam) dataset with comprehensive analysis\n",
                "\n",
                "This notebook includes:\n",
                "- ✅ 12 different experiment configurations\n",
                "- ✅ Marathi NER dataset from HuggingFace (ai4bharat/naamapadam)\n",
                "- ✅ Heatmap generation for receptor and glomeruli activations\n",
                "- ✅ Comprehensive metrics (micro/macro averages, per-entity F1)\n",
                "- ✅ Model saving and comparison\n",
                "- ✅ Visualization and results export\n",
                "\n",
                "**Estimated Runtime**: 6-10 hours on GPU (T4), longer on CPU (Marathi dataset is larger)\n",
                "\n",
                "---"
            ]
    
    # Update code cells to use Marathi dataset
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_str = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Update the paths in all experiment cells
        if '--config config/tuning_experiments.yaml' in source_str:
            # This is an experiment cell - need to update to use Marathi config
            for exp_name in ['activation_gelu', 'activation_swish', 'activation_mish', 
                            'receptors_256', 'receptors_64', 'glomeruli_256', 'glomeruli_64',
                            'dropout_03', 'dropout_01', 'learning_rate_0005', 'learning_rate_00005',
                            'batch_size_64']:
                if f"--experiment {exp_name}" in source_str:
                    # Update to use Marathi config
                    new_source = source_str.replace(
                        'config/tuning_experiments.yaml',
                        'config/tuning_experiments_marathi.yaml'
                    ).replace(
                        f'results/{exp_name}',
                        f'results_marathi/{exp_name}'
                    )
                    cell['source'] = new_source.split('\n')
                    break

# Clear all outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []

# Save the new notebook
with open(r'notebooks\comprehensive_experiments_marathi.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Successfully created comprehensive_experiments_marathi.ipynb")
