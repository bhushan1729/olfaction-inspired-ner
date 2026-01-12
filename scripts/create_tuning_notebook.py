#!/usr/bin/env python3
"""
Create complete hyperparameter tuning notebook for CoNLL-2003.
All 8 experiments included.
"""

import json

def create_notebook():
    """Build the complete notebook structure."""
    
    cells = []
    
    # Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": "# 🔬 Comprehensive Hyperparameter Tuning\\n\\n**Dataset**: CoNLL-2003\\n**Experiments**: 8 systematic variations\\n\\n1. Baseline (ReLU)\\n2. Baseline (GELU)\\n3. More Receptors (256)\\n4. More Glomeruli (64)\\n5. Larger LSTM (512)\\n6. Lower Dropout (0.2)\\n7. Larger Batch (64)\\n8. Strong Regularization\\n\\n**Each experiment auto**:\\n- Trains & evaluates\\n- Saves model → Google Drive\\n- Saves results → GitHub\\n- Generates curves & plots"
    })
    
    # Setup cells
    cells.extend([
        {"cell_type": "markdown", "metadata": {}, "source": "## Setup"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "import torch\\nprint(f'PyTorch: {torch.__version__}')\\nprint(f'CUDA: {torch.cuda.is_available()}')\\nif torch.cuda.is_available():\\n    print(f'GPU: {torch.cuda.get_device_name(0)}')"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "!git clone https://github.com/bhushan1729/olfaction-inspired-ner.git\\n%cd olfaction-inspired-ner"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "!pip install -q torch numpy scikit-learn seqeval matplotlib seaborn pandas tqdm tensorboard"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "import os\\nif not os.path.exists('./data/glove.6B.300d.txt'):\\n    !mkdir -p data\\n    !wget -q http://nlp.stanford.edu/data/glove.6B.zip -O data/glove.6B.zip\\n    !unzip -q data/glove.6B.zip -d data/\\n    !rm data/glove.6B.zip"},
        {"cell_type": "markdown", "metadata": {}, "source": "## Google Drive"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "from google.colab import drive\\ndrive.mount('/content/drive')\\n\\nmodel_dir = '/content/drive/MyDrive/olfaction_ner/models'\\n!mkdir -p {model_dir}"},
        {"cell_type": "markdown", "metadata": {}, "source": "## Helpers"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "import json, shutil, matplotlib.pyplot as plt\\nfrom pathlib import Path\\n\\n!mkdir -p comparison_plots experiment_results/CoNLL-2003\\nall_experiments = []\\n\\ndef save_to_drive(name, save_dir):\\n    dst = f'/content/drive/MyDrive/olfaction_ner/models/{name}.pt'\\n    shutil.copy(f'{save_dir}/best_model.pt', dst)\\n    print(f'✓ Drive: {dst}')\\n\\ndef save_results(name, cfg, res, sd):\\n    ed = f'experiment_results/CoNLL-2003/{name}'\\n    Path(ed).mkdir(parents=True, exist_ok=True)\\n    with open(f'{ed}/metadata.json', 'w') as f:\\n        json.dump({'name': name, 'config': cfg}, f, indent=2)\\n    with open(f'{ed}/results.json', 'w') as f:\\n        json.dump(res, f, indent=2)\\n    all_experiments.append({'name': name, 'config': cfg, 'results': res})\\n\\ndef plot_curves(name, res, path):\\n    fig, ax = plt.subplots(1, 2, figsize=(14, 5))\\n    epochs = [e['epoch'] for e in res['epochs']]\\n    ax[0].plot(epochs, [e['train']['total_loss'] for e in res['epochs']])\\n    ax[0].set_title(f'{name}: Loss')\\n    ax[1].plot(epochs, [e['valid']['f1'] for e in res['epochs']], color='green')\\n    ax[1].set_title(f'{name}: F1')\\n    plt.tight_layout()\\n    plt.savefig(path, dpi=150, bbox_inches='tight')\\n    plt.close()\\nprint('✓ Ready')"},
    ])
    
    # Experiments
    experiments = [
        ("exp1_relu", "Baseline (ReLU)", "config/experiments.yaml", "olfactory_full",
         "{'activation':'relu','receptors':128,'glomeruli':32,'lstm':256,'dropout':0.5,'batch':32,'sparse':0.001,'diverse':0.01}"),
        ("exp2_gelu", "Baseline (GELU)", "config/tuning_experiments.yaml", "activation_gelu",
         "{'activation':'gelu','receptors':128,'glomeruli':32,'lstm':256,'dropout':0.5,'batch':32,'sparse':0.001,'diverse':0.01}"),
        ("exp3_rec256", "More Receptors", "config/hyperparameter_tuning.yaml", "exp3_more_receptors",
         "{'activation':'gelu','receptors':256,'glomeruli':64,'lstm':256,'dropout':0.5,'batch':32,'sparse':0.001,'diverse':0.01}"),
        ("exp4_glom64", "More Glomeruli", "config/hyperparameter_tuning.yaml", "exp4_more_glomeruli",
         "{'activation':'gelu','receptors':128,'glomeruli':64,'lstm':256,'dropout':0.5,'batch':32,'sparse':0.001,'diverse':0.01}"),
        ("exp5_lstm512", "Larger LSTM", "config/hyperparameter_tuning.yaml", "exp5_larger_lstm",
         "{'activation':'gelu','receptors':128,'glomeruli':32,'lstm':512,'dropout':0.5,'batch':32,'sparse':0.001,'diverse':0.01}"),
        ("exp6_drop02", "Lower Dropout", "config/hyperparameter_tuning.yaml", "exp6_lower_dropout",
         "{'activation':'gelu','receptors':128,'glomeruli':32,'lstm':256,'dropout':0.2,'batch':32,'sparse':0.001,'diverse':0.01}"),
        ("exp7_batch64", "Larger Batch", "config/hyperparameter_tuning.yaml", "exp7_larger_batch",
         "{'activation':'gelu','receptors':128,'glomeruli':32,'lstm':256,'dropout':0.5,'batch':64,'sparse':0.001,'diverse':0.01}"),
        ("exp8_strongreg", "Strong Reg", "config/hyperparameter_tuning.yaml", "exp8_strong_reg",
         "{'activation':'gelu','receptors':128,'glomeruli':32,'lstm':256,'dropout':0.5,'batch':32,'sparse':0.1,'diverse':0.1}"),
    ]
    
    cells.append({"cell_type": "markdown", "metadata": {}, "source": "---\\n# Experiments\\n---"})
    
    for i, (name, title, cfg_file, cfg_name, cfg_dict) in enumerate(experiments, 1):
        cells.extend([
            {"cell_type": "markdown", "metadata": {}, "source": f"## Experiment {i}: {title}"},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], 
             "source": f"!python src/train.py --config {cfg_file} --experiment {cfg_name} --save_dir results/tuning/{name}"},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": f"with open('results/tuning/{name}/results.json') as f:\\n    results = json.load(f)\\n\\ncfg = {cfg_dict}\\nsave_to_drive('{name}', 'results/tuning/{name}')\\nsave_results('{name}', cfg, results, 'results/tuning/{name}')\\nplot_curves('{title}', results, 'comparison_plots/{name}.png')\\nprint(f\\\"✅ Exp{i}: F1={{results['test']['f1']:.4f}}\\\")"},
        ])
    
    # Comparison
    cells.extend([
        {"cell_type": "markdown", "metadata": {}, "source": "---\\n# Comparison\\n---"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": "import pandas as pd\\n\\ndata = []\\nfor e in all_experiments:\\n    data.append({\\n        'Name': e['name'],\\n        'F1': f\\\"{e['results']['test']['f1']:.4f}\\\",\\n        'Precision': f\\\"{e['results']['test']['precision']:.4f}\\\",\\n        'Recall': f\\\"{e['results']['test']['recall']:.4f}\\\"\\n    })\\n\\ndf = pd.DataFrame(data)\\nprint(df.to_string(index=False))\\ndf.to_csv('comparison_plots/results.csv', index=False)"},
        {"cell_type": "markdown", "metadata": {}, "source": "## Push to GitHub"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": "!git config user.name \\\"Colab\\\"\\n!git config user.email \\\"colab@exp.local\\\"\\n!git add experiment_results/ comparison_plots/\\n!git commit -m \\\"Hyperparameter tuning results\\\"\\n!git push origin main\\nprint('✅ Pushed to GitHub')"},
    ])
    
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU"},
        "nbformat": 4,
        "nbformat_minor": 4
    }

if __name__ == "__main__":
    notebook = create_notebook()
    with open('notebooks/hyperparameter_tuning_conll2003.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"✓ Created notebook with {len(notebook['cells'])} cells")
    print("✓ All 8 experiments included")
    print("✓ Ready to run in Colab")
