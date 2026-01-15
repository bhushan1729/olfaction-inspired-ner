#!/usr/bin/env python3
"""
Monitor the progress of running experiments.
"""

import os
import json
import time
from pathlib import Path

def check_experiment_status():
    """Check which experiments have completed."""
    results_dir = Path('results')
    experiment_results_dir = Path('experiment_results/CoNLL-2003')
    
    expected_experiments = [
        'baseline', 'olfactory_full', 'olfactory_no_sparse', 'olfactory_no_glomeruli',
        'exp3_more_receptors', 'exp4_more_glomeruli', 'exp5_larger_lstm',
        'exp6_lower_dropout', 'exp7_larger_batch', 'exp8_strong_reg', 'activation_gelu'
    ]
    
    completed = []
    in_progress = []
    not_started = []
    
    for exp_name in expected_experiments:
        results_file = results_dir / exp_name / 'results.json'
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                if 'test' in results:
                    completed.append((exp_name, results['test']['f1']))
                else:
                    in_progress.append(exp_name)
            except:
                in_progress.append(exp_name)
        else:
            not_started.append(exp_name)
    
    return completed, in_progress, not_started

def main():
    """Main monitoring function."""
    print("=" * 80)
    print("EXPERIMENT MONITOR")
    print("=" * 80)
    
    completed, in_progress, not_started = check_experiment_status()
    
    print(f"\n✓ Completed: {len(completed)}/11")
    for name, f1 in completed:
        print(f"  - {name}: F1={f1:.4f}")
    
    print(f"\n⏳ In Progress: {len(in_progress)}/11")
    for name in in_progress:
        print(f"  - {name}")
    
    print(f"\n⏸  Not Started: {len(not_started)}/11")
    for name in not_started:
        print(f"  - {name}")
    
    print("\n" + "=" * 80)
    
    # Check log file
    if os.path.exists('experiment_run.log'):
        print("\nLast 20 lines of log:")
        print("-" * 80)
        os.system('tail -20 experiment_run.log')

if __name__ == '__main__':
    main()
