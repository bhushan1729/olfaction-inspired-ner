"""
Import existing experiment results into tracking system.
Run this once to populate experiment_results/ with completed experiments.
"""

import json
import os
import shutil
from pathlib import Path
from src.utils.save_results import save_experiment_results, generate_results_index


def import_conll2003_results():
    """Import all CoNLL-2003 experiments."""
    
    # Baseline
    baseline_config = {
        'model_type': 'baseline',
        'embed_dim': 300,
        'lstm_hidden': 256,
        'lstm_layers': 1,
        'dropout': 0.5,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    baseline_results = {
        'test': {
            'f1': 0.7730,
            'precision': 0.8008,
            'recall': 0.7472,
            'per_entity': {
                'LOC': 0.8528,
                'MISC': 0.6784,
                'ORG': 0.7340,
                'PER': 0.7697
            }
        },
        'best_f1': 0.8509,
        'epochs': [{'epoch': 17}]
    }
    
    save_experiment_results(
        experiment_name='baseline_conll2003',
        dataset_name='CoNLL-2003',
        model_type='baseline',
        config=baseline_config,
        results=baseline_results
    )
    
    # Olfactory (ReLU)
    relu_config = {
        'model_type': 'olfactory',
        'embed_dim': 300,
        'num_receptors': 128,
        'num_glomeruli': 32,
        'receptor_activation': 'relu',
        'lstm_hidden': 256,
        'lstm_layers': 1,
        'dropout': 0.5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'lambda_sparse': 0.001,
        'lambda_diverse': 0.01
    }
    
    relu_results = {
        'test': {
            'f1': 0.7256,
            'precision': 0.7679,
            'recall': 0.6877,
            'per_entity': {
                'LOC': 0.8138,
                'MISC': 0.6227,
                'ORG': 0.6899,
                'PER': 0.7120
            }
        },
        'best_f1': 0.7256
    }
    
    save_experiment_results(
        experiment_name='olfactory_relu_conll2003',
        dataset_name='CoNLL-2003',
        model_type='olfactory_relu',
        config=relu_config,
        results=relu_results
    )
    
    # Olfactory (GELU)
    gelu_config = {
        'model_type': 'olfactory',
        'embed_dim': 300,
        'num_receptors': 128,
        'num_glomeruli': 32,
        'receptor_activation': 'gelu',
        'lstm_hidden': 256,
        'lstm_layers': 1,
        'dropout': 0.5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'lambda_sparse': 0.001,
        'lambda_diverse': 0.01
    }
    
    gelu_results = {
        'test': {
            'f1': 0.7306,
            'precision': 0.7873,
            'recall': 0.6815,
            'per_entity': {
                'LOC': 0.8082,
                'MISC': 0.6761,
                'ORG': 0.7067,
                'PER': 0.6914
            }
        },
        'best_f1': 0.7306,
        'epochs': [{'epoch': 22}]
    }
    
    save_experiment_results(
        experiment_name='olfactory_gelu_conll2003',
        dataset_name='CoNLL-2003',
        model_type='olfactory_gelu',
        config=gelu_config,
        results=gelu_results
    )
    
    # Olfactory (No Glomeruli)
    no_glom_config = {
        'model_type': 'olfactory',
        'embed_dim': 300,
        'num_receptors': 128,
        'num_glomeruli': 32,
        'receptor_activation': 'relu',
        'lstm_hidden': 256,
        'lstm_layers': 1,
        'dropout': 0.5,
        'use_receptors': True,
        'use_glomeruli': False,
        'batch_size': 32,
        'learning_rate': 0.001,
        'lambda_sparse': 0.001,
        'lambda_diverse': 0.01
    }
    
    no_glom_results = {
        'test': {
            'f1': 0.7353,
            'precision': 0.7878,
            'recall': 0.6888,
            'per_entity': {
                'LOC': 0.8198,
                'MISC': 0.6346,
                'ORG': 0.6863,
                'PER': 0.7212
            }
        },
        'best_f1': 0.7353
    }
    
    save_experiment_results(
        experiment_name='olfactory_no_glomeruli_conll2003',
        dataset_name='CoNLL-2003',
        model_type='olfactory_no_glomeruli',
        config=no_glom_config,
        results=no_glom_results
    )
    
    print("\n✓ Imported 4 CoNLL-2003 experiments")


def copy_visualizations():
    """Copy existing visualizations to experiment_results."""
    
    src_docs = Path('./docs')
    
    # GELU visualizations
    gelu_dest = Path('./experiment_results/CoNLL-2003/olfactory_gelu_conll2003/visualizations')
    gelu_dest.mkdir(parents=True, exist_ok=True)
    
    if (src_docs / 'gelu_heatmap.png').exists():
        shutil.copy(src_docs / 'gelu_heatmap.png', gelu_dest / 'receptor_heatmap.png')
    if (src_docs / 'gelu_tsne.png').exists():
        shutil.copy(src_docs / 'gelu_tsne.png', gelu_dest / 'glomeruli_tsne.png')
    if (src_docs / 'model_comparison.png').exists():
        shutil.copy(src_docs / 'model_comparison.png', gelu_dest / 'model_comparison.png')
    
    print("✓ Copied GELU visualizations")


if __name__ == '__main__':
    print("Importing existing experiment results...\n")
    
    import_conll2003_results()
    copy_visualizations()
    
    # Generate index
    from src.utils.save_results import generate_results_index
    generate_results_index()
    
    print("\n✅ All results imported!")
    print("\nView at: experiment_results/README.md")
