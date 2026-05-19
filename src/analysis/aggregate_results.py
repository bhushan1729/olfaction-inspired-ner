import os
import json
import numpy as np
import argparse
from collections import defaultdict

def aggregate_results(base_dir):
    """
    Scans the results directory and aggregates metrics across multiple seeds.
    Reports Mean ± Std for F1, Precision, and Recall.
    """
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    print(f"Aggregating results from: {base_dir}\n")

    # Collect all results across dynamic structures using os.walk
    # grouped_results: { (dataset_lang, experiment): { 'f1': [], 'precision': [], 'recall': [], 'seeds': 0 } }
    grouped_results = defaultdict(lambda: {
        'f1': [],
        'precision': [],
        'recall': [],
        'seeds': 0
    })

    # Keep track of found/skipped files for diagnostics
    found_results_files = []
    skipped_paths = []
    unparseable_paths = []
    error_paths = []

    # Known list of experiments
    known_experiments = {
        'baseline', 'olfactory', 'receptors_only', 'no_sparsity', 
        'more_receptors', 'more_glomeruli', 'more_receptors_more_glomeruli'
    }
    
    # Known list of datasets
    known_datasets = {
        'conll_en', 'conll_en_1k', 'wikiann_mr', 'wikiann_mr_1k', 
        'wikiann_hi', 'wikiann_hi_1k', 'wikiann_ta', 'wikiann_ta_1k', 
        'wikiann_bn', 'wikiann_bn_1k', 'wikiann_te', 'wikiann_te_1k'
    }

    for root, dirs, files in os.walk(base_dir):
        if 'results.json' in files:
            results_file = os.path.join(root, 'results.json')
            found_results_files.append(results_file)
            
            # Normalize path to forward slashes to prevent OS mismatch issues
            rel_path = os.path.relpath(root, base_dir)
            parts = [p for p in rel_path.replace('\\', '/').split('/') if p and p != '.']
            
            # Robust extraction of dataset, experiment, seed
            dataset_lang = "unknown_dataset"
            experiment = "unknown_experiment"
            seed = "unknown_seed"
            
            remaining_parts = []
            for part in parts:
                part_lower = part.lower()
                if part_lower.startswith('seed_') or part_lower.isdigit():
                    seed = part
                elif part_lower in known_experiments:
                    experiment = part
                elif part_lower in known_datasets:
                    dataset_lang = part
                else:
                    remaining_parts.append(part)
            
            # Position-based heuristics if some parts are still unknown
            if dataset_lang == "unknown_dataset" and len(parts) >= 1:
                dataset_lang = parts[0]
            
            if experiment == "unknown_experiment":
                # Check standard layouts:
                # 1. dataset/experiment/seed -> parts length 3
                if len(parts) == 3:
                    experiment = parts[1]
                # 2. dataset/lang/experiment/seed -> parts length 4
                elif len(parts) == 4:
                    experiment = parts[2]
                elif remaining_parts:
                    experiment = remaining_parts[-1]
            
            # If we still can't identify experiment or dataset, track as unparseable but try to group anyway
            if dataset_lang == "unknown_dataset" or experiment == "unknown_experiment":
                unparseable_paths.append((rel_path, dataset_lang, experiment, seed))
            
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    test_metrics = data.get('test', {})
                    
                    # Also try to support older/alternative keys like test_f1
                    f1 = test_metrics.get('f1', test_metrics.get('test_f1'))
                    precision = test_metrics.get('precision', test_metrics.get('test_precision'))
                    recall = test_metrics.get('recall', test_metrics.get('test_recall'))
                    
                    if f1 is None and 'f1-score' in test_metrics:
                        f1 = test_metrics['f1-score']
                    
                    key = (dataset_lang, experiment)
                    metrics_added = False
                    
                    if f1 is not None:
                        # If F1 is already 0-100, don't multiply by 100
                        val_f1 = f1 * 100 if f1 <= 1.0 else f1
                        grouped_results[key]['f1'].append(val_f1)
                        metrics_added = True
                    if precision is not None:
                        val_p = precision * 100 if precision <= 1.0 else precision
                        grouped_results[key]['precision'].append(val_p)
                        metrics_added = True
                    if recall is not None:
                        val_r = recall * 100 if recall <= 1.0 else recall
                        grouped_results[key]['recall'].append(val_r)
                        metrics_added = True
                        
                    if metrics_added:
                        grouped_results[key]['seeds'] += 1
                    else:
                        skipped_paths.append((rel_path, "No valid test metrics found in results.json"))
            except Exception as e:
                error_paths.append((rel_path, str(e)))

    # Print results table
    active_keys = [k for k in grouped_results.keys() if grouped_results[k]['seeds'] > 0]
    
    if active_keys:
        print("="*110)
        print(f"{'Dataset':<15} | {'Experiment':<15} | {'F1 (Mean ± SD)':<18} | {'Precision':<18} | {'Recall':<18} | {'Seeds'}")
        print("="*110)
        for key in sorted(active_keys):
            dataset_lang, experiment = key
            metrics = grouped_results[key]
            seeds_found = metrics['seeds']
            
            mean_f1 = np.mean(metrics['f1']) if metrics['f1'] else 0
            std_f1 = np.std(metrics['f1']) if metrics['f1'] else 0
            mean_p = np.mean(metrics['precision']) if metrics['precision'] else 0
            std_p = np.std(metrics['precision']) if metrics['precision'] else 0
            mean_r = np.mean(metrics['recall']) if metrics['recall'] else 0
            std_r = np.std(metrics['recall']) if metrics['recall'] else 0
            
            print(f"{dataset_lang:<15} | {experiment:<15} | {mean_f1:>5.2f} ± {std_f1:<4.2f}% | {mean_p:>5.2f} ± {std_p:<4.2f}% | {mean_r:>5.2f} ± {std_r:<4.2f}% | {seeds_found}")
        print("="*110)
    else:
        print("⚠ No valid results were aggregated.")
        print("\n=== DIAGNOSTICS & TROUBLESHOOTING ===")
        print(f"Total 'results.json' files found: {len(found_results_files)}")
        
        if found_results_files:
            print("\nFirst 5 results files found:")
            for rf in found_results_files[:5]:
                print(f"  - {os.path.relpath(rf, base_dir)}")
                
            if unparseable_paths:
                print("\nPaths that did not fit standard layout:")
                for up, ds, exp, sd in unparseable_paths[:5]:
                    print(f"  - Path: {up} (Parsed as Dataset: {ds}, Experiment: {exp}, Seed: {sd})")
                    
            if skipped_paths:
                print("\nSkipped files (e.g. missing metrics):")
                for sp, reason in skipped_paths[:5]:
                    print(f"  - {sp}: {reason}")
                    
            if error_paths:
                print("\nError loading these files:")
                for ep, err in error_paths[:5]:
                    print(f"  - {ep}: {err}")
        else:
            print("\nCould not find any 'results.json' files in the specified directory.")
            print("Please make sure:")
            print("  1. The path to your low-resource experiments folder is correct.")
            print("  2. Your experiments have run and completed successfully.")
            print("  3. The files are named 'results.json'.")
            
            # List immediate subdirectories in base_dir
            try:
                subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                print(f"\nSubdirectories found in base directory:")
                for sd in sorted(subdirs)[:10]:
                    print(f"  - {sd}")
            except Exception as e:
                print(f"Could not list directory: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate NER results across seeds')
    parser.add_argument('--dir', type=str, default='/content/drive/My Drive/olfaction_inspired_ner', help='Base results directory')
    args = parser.parse_args()
    
    aggregate_results(args.dir)
