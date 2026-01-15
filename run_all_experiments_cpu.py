#!/usr/bin/env python3
"""
Sequential experiment runner for olfaction-inspired NER.
Runs all experiments on CPU without Ray cluster.
"""

import os
import sys
import yaml
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.train import train, load_config


def load_all_experiments():
    """Load all experiment configurations."""
    experiments = {}
    
    # Load from experiments.yaml
    with open('config/experiments.yaml', 'r') as f:
        exp_config = yaml.safe_load(f)
    
    experiments['baseline'] = exp_config['baseline']
    experiments['olfactory_full'] = exp_config['olfactory_full']
    experiments['olfactory_no_sparse'] = exp_config['olfactory_no_sparse']
    experiments['olfactory_no_glomeruli'] = exp_config['olfactory_no_glomeruli']
    
    # Merge data and training configs
    data_config = exp_config.get('data', {})
    training_config = exp_config.get('training', {})
    
    for name in experiments:
        experiments[name].update(data_config)
        experiments[name].update(training_config)
        experiments[name]['device'] = 'cpu'  # Force CPU
    
    # Load from hyperparameter_tuning.yaml
    with open('config/hyperparameter_tuning.yaml', 'r') as f:
        tuning_config = yaml.safe_load(f)
    
    tuning_experiments = {
        'exp3_more_receptors': tuning_config['exp3_more_receptors'],
        'exp4_more_glomeruli': tuning_config['exp4_more_glomeruli'],
        'exp5_larger_lstm': tuning_config['exp5_larger_lstm'],
        'exp6_lower_dropout': tuning_config['exp6_lower_dropout'],
        'exp7_larger_batch': tuning_config['exp7_larger_batch'],
        'exp8_strong_reg': tuning_config['exp8_strong_reg'],
    }
    
    # Merge data and training configs for tuning experiments
    tuning_data_config = tuning_config.get('data', {})
    tuning_training_config = tuning_config.get('training', {})
    
    for name, config in tuning_experiments.items():
        config.update(tuning_data_config)
        config.update(tuning_training_config)
        config['device'] = 'cpu'  # Force CPU
        experiments[name] = config
    
    # Add GELU activation experiment (same as olfactory_full but with GELU)
    experiments['activation_gelu'] = experiments['olfactory_full'].copy()
    experiments['activation_gelu']['receptor_activation'] = 'gelu'
    
    return experiments


def plot_training_curves(results, save_path):
    """Plot training curves for an experiment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = [e['epoch'] for e in results['epochs']]
    train_losses = [e['train']['total_loss'] for e in results['epochs']]
    valid_f1s = [e['valid']['f1'] for e in results['epochs']]
    
    # Loss curve
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # F1 curve
    axes[1].plot(epochs, valid_f1s, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation F1', fontsize=12)
    axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_plots(all_results, save_dir):
    """Create comparison plots across all experiments."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data
    exp_names = []
    test_f1s = []
    test_precisions = []
    test_recalls = []
    
    for name, results in all_results.items():
        if 'test' in results:
            exp_names.append(name)
            test_f1s.append(results['test']['f1'])
            test_precisions.append(results['test']['precision'])
            test_recalls.append(results['test']['recall'])
    
    # Overall metrics comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(exp_names))
    width = 0.25
    
    ax.bar([i - width for i in x], test_f1s, width, label='F1', alpha=0.8)
    ax.bar([i for i in x], test_precisions, width, label='Precision', alpha=0.8)
    ax.bar([i + width for i in x], test_recalls, width, label='Recall', alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test Set Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_all_experiments.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Per-entity F1 comparison
    entity_data = {}
    for name, results in all_results.items():
        if 'test' in results and 'per_entity' in results['test']:
            for entity, f1 in results['test']['per_entity'].items():
                if entity not in entity_data:
                    entity_data[entity] = {}
                entity_data[entity][name] = f1
    
    if entity_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        entities = list(entity_data.keys())
        x = range(len(entities))
        width = 0.8 / len(exp_names)
        
        for i, exp_name in enumerate(exp_names):
            f1_scores = [entity_data[entity].get(exp_name, 0) for entity in entities]
            ax.bar([xi + i * width for xi in x], f1_scores, width, label=exp_name, alpha=0.8)
        
        ax.set_xlabel('Entity Type', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Per-Entity F1 Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks([xi + width * len(exp_names) / 2 for xi in x])
        ax.set_xticklabels(entities, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_entity_f1_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()


def save_metadata(experiment_name, config, results, save_dir):
    """Save experiment metadata."""
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'test_f1': results.get('test', {}).get('f1', 0),
        'test_precision': results.get('test', {}).get('precision', 0),
        'test_recall': results.get('test', {}).get('recall', 0),
        'best_validation_f1': results.get('best_f1', 0),
        'total_epochs': len(results.get('epochs', [])),
    }
    
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main execution function."""
    print("=" * 80)
    print("OLFACTION-INSPIRED NER - SEQUENTIAL EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Running on: CPU")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load all experiments
    print("\nLoading experiment configurations...")
    experiments = load_all_experiments()
    print(f"Found {len(experiments)} experiments to run:")
    for i, name in enumerate(experiments.keys(), 1):
        print(f"  {i}. {name}")
    
    # Create output directories
    os.makedirs('experiment_results/CoNLL-2003', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run each experiment
    all_results = {}
    
    for i, (exp_name, config) in enumerate(experiments.items(), 1):
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp_name}")
        print("=" * 80)
        
        # Set up save directories
        results_dir = f'results/{exp_name}'
        exp_results_dir = f'experiment_results/CoNLL-2003/{exp_name}_conll2003'
        
        # Check if already completed
        if os.path.exists(os.path.join(results_dir, 'results.json')):
            print(f"⚠️  Experiment {exp_name} already completed. Loading existing results...")
            with open(os.path.join(results_dir, 'results.json'), 'r') as f:
                results = json.load(f)
            all_results[exp_name] = results
            print(f"✓ Loaded existing results (Test F1: {results.get('test', {}).get('f1', 0):.4f})")
            continue
        
        try:
            # Run training
            results = train(config, exp_name, results_dir)
            all_results[exp_name] = results
            
            # Create experiment results directory
            os.makedirs(exp_results_dir, exist_ok=True)
            
            # Copy results to experiment_results
            with open(os.path.join(exp_results_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save metadata
            save_metadata(exp_name, config, results, exp_results_dir)
            
            # Plot training curves
            plot_training_curves(results, os.path.join(exp_results_dir, 'training_curves.png'))
            
            # Copy best model if it exists
            model_path = os.path.join(results_dir, 'best_model.pt')
            if os.path.exists(model_path):
                import shutil
                shutil.copy(model_path, os.path.join(exp_results_dir, 'best_model.pt'))
            
            print(f"\n✓ Experiment {exp_name} completed successfully!")
            print(f"  Test F1: {results['test']['f1']:.4f}")
            print(f"  Test Precision: {results['test']['precision']:.4f}")
            print(f"  Test Recall: {results['test']['recall']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error in experiment {exp_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison plots
    print("\n" + "=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    create_comparison_plots(all_results, 'visualizations')
    print("✓ Comparison plots saved to visualizations/")
    
    # Save summary
    summary = {
        'total_experiments': len(experiments),
        'completed_experiments': len(all_results),
        'timestamp': datetime.now().isoformat(),
        'results': {
            name: {
                'test_f1': results.get('test', {}).get('f1', 0),
                'test_precision': results.get('test', {}).get('precision', 0),
                'test_recall': results.get('test', {}).get('recall', 0),
            }
            for name, results in all_results.items()
        }
    }
    
    with open('experiment_results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"Completed: {len(all_results)}/{len(experiments)} experiments")
    print(f"Results saved to: experiment_results/")
    print(f"Visualizations saved to: visualizations/")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
