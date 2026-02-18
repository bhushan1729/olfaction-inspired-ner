"""
Orchestrate baseline vs olfactory experiments across multiple datasets.

This script runs the minimal, correct experiment to validate the hypothesis:
"Does my biologically inspired olfactory feature extractor add value beyond 
a standard language model for NER?"
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Experiment configurations
DATASETS = {
    'conll2003': {
        'dataset': 'conll2003',
        'language': None,
        'display_name': 'CoNLL-2003 (English)'
    },
    'hindi': {
        'dataset': 'wikiann',
        'language': 'hi',
        'display_name': 'WikiANN Hindi'
    },
    'marathi': {
        'dataset': 'wikiann',
        'language': 'mr',
        'display_name': 'WikiANN Marathi'
    },
    'tamil': {
        'dataset': 'wikiann',
        'language': 'ta',
        'display_name': 'WikiANN Tamil'
    },
    'bangla': {
        'dataset': 'wikiann',
        'language': 'bn',
        'display_name': 'WikiANN Bangla'
    },
    'telugu': {
        'dataset': 'wikiann',
        'language': 'te',
        'display_name': 'WikiANN Telugu'
    }
}


def run_experiment(dataset_key, experiment_type, epochs, batch_size, lr, save_dir):
    """
    Run a single experiment.
    
    Args:
        dataset_key: Key in DATASETS dict
        experiment_type: 'baseline' or 'olfactory'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        save_dir: Directory to save results
    
    Returns:
        Success status and path to results
    """
    dataset_info = DATASETS[dataset_key]
    
    print(f"\n{'='*80}")
    print(f"Running {experiment_type.upper()} on {dataset_info['display_name']}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        'src/train_bert.py',
        '--dataset', dataset_info['dataset'],
        '--experiment', experiment_type,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--save_dir', save_dir
    ]
    
    if dataset_info['language']:
        cmd.extend(['--language', dataset_info['language']])
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Find results file
        lang = dataset_info['language'] if dataset_info['language'] else 'en'
        results_path = Path(save_dir) / dataset_info['dataset'] / lang / f"mbert_{experiment_type}" / "results.json"
        
        if results_path.exists():
            print(f"✓ Experiment completed successfully")
            print(f"  Results saved to: {results_path}")
            return True, str(results_path)
        else:
            print(f"✗ Results file not found: {results_path}")
            return False, None
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment failed with error:")
        print(e.stderr)
        return False, None


def run_all_experiments(args):
    """Run all baseline and olfactory experiments."""
    
    # Select datasets
    if args.datasets:
        dataset_keys = [k for k in args.datasets.split(',') if k in DATASETS]
    else:
        dataset_keys = list(DATASETS.keys())
    
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE: Running 1 epoch on CoNLL-2003 only\n")
        dataset_keys = ['conll2003']
        epochs = 1
    else:
        epochs = args.epochs
    
    # Track results
    results_summary = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'epochs': epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'datasets': dataset_keys
        },
        'experiments': {}
    }
    
    # Run experiments
    for dataset_key in dataset_keys:
        dataset_name = DATASETS[dataset_key]['display_name']
        results_summary['experiments'][dataset_name] = {}
        
        # Run baseline
        success, baseline_path = run_experiment(
            dataset_key, 'baseline', epochs, 
            args.batch_size, args.lr, args.save_dir
        )
        results_summary['experiments'][dataset_name]['baseline'] = {
            'success': success,
            'results_path': baseline_path
        }
        
        # Run olfactory
        success, olfactory_path = run_experiment(
            dataset_key, 'olfactory', epochs,
            args.batch_size, args.lr, args.save_dir
        )
        results_summary['experiments'][dataset_name]['olfactory'] = {
            'success': success,
            'results_path': olfactory_path
        }
    
    # Save summary
    results_summary['end_time'] = datetime.now().isoformat()
    summary_path = Path(args.save_dir) / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")
    
    # Print quick summary
    print("\nResults Summary:")
    print(f"{'Dataset':<30} {'Baseline':<15} {'Olfactory':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for dataset_name, exp_results in results_summary['experiments'].items():
        baseline_path = exp_results['baseline']['results_path']
        olfactory_path = exp_results['olfactory']['results_path']
        
        if baseline_path and olfactory_path and Path(baseline_path).exists() and Path(olfactory_path).exists():
            with open(baseline_path) as f:
                baseline_metrics = json.load(f)
            with open(olfactory_path) as f:
                olfactory_metrics = json.load(f)
            
            baseline_f1 = baseline_metrics.get('test_f1', 0.0)
            olfactory_f1 = olfactory_metrics.get('test_f1', 0.0)
            improvement = olfactory_f1 - baseline_f1
            
            print(f"{dataset_name:<30} {baseline_f1:<15.4f} {olfactory_f1:<15.4f} {improvement:+.4f}")
        else:
            print(f"{dataset_name:<30} {'FAILED':<15} {'FAILED':<15} {'N/A':<15}")
    
    print("\nNext steps:")
    print("  1. Run: python src/analysis/compare_results.py --results_dir", args.save_dir)
    print("  2. Review comparative analysis and visualizations")
    
    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run baseline vs olfactory NER experiments'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Comma-separated list of datasets (e.g., "conll2003,hindi"). Default: all'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./results',
        help='Directory to save results (default: ./results)'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Quick test mode: 1 epoch on CoNLL-2003 only'
    )
    
    args = parser.parse_args()
    
    # Run experiments
    run_all_experiments(args)
