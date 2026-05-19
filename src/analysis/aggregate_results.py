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

    # Structure is base_dir / dataset / language / experiment / seed_X / results.json
    print(f"Aggregating results from: {base_dir}\n")
    print("="*110)
    print(f"{'Dataset':<15} | {'Experiment':<15} | {'F1 (Mean ± SD)':<18} | {'Precision':<18} | {'Recall':<18} | {'Seeds'}")
    print("="*110)

    # Collect all results across dynamic structures using os.walk
    # grouped_results: { (dataset_lang, experiment): { 'f1': [], 'precision': [], 'recall': [], 'seeds': 0 } }
    grouped_results = defaultdict(lambda: {
        'f1': [],
        'precision': [],
        'recall': [],
        'seeds': 0
    })

    for root, dirs, files in os.walk(base_dir):
        if 'results.json' in files:
            results_file = os.path.join(root, 'results.json')
            rel_path = os.path.relpath(root, base_dir)
            parts = rel_path.split(os.sep)
            
            # We expect either 3 parts (dataset_key, experiment, seed)
            # or 4 parts (dataset, language, experiment, seed)
            if len(parts) == 3:
                dataset_key = parts[0]
                experiment = parts[1]
                seed = parts[2]
                dataset_lang = dataset_key
            elif len(parts) == 4:
                dataset = parts[0]
                language = parts[1]
                experiment = parts[2]
                seed = parts[3]
                dataset_lang = f"{dataset}_{language}" if language != 'default' else dataset
            else:
                # Unsupported nested directory structure, skip
                continue

            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    test_metrics = data.get('test', {})
                    
                    key = (dataset_lang, experiment)
                    if 'f1' in test_metrics:
                        grouped_results[key]['f1'].append(test_metrics['f1'] * 100)
                    if 'precision' in test_metrics:
                        grouped_results[key]['precision'].append(test_metrics['precision'] * 100)
                    if 'recall' in test_metrics:
                        grouped_results[key]['recall'].append(test_metrics['recall'] * 100)
                    grouped_results[key]['seeds'] += 1
            except Exception as e:
                pass

    # Print results
    for key in sorted(grouped_results.keys()):
        dataset_lang, experiment = key
        metrics = grouped_results[key]
        seeds_found = metrics['seeds']
        
        if seeds_found > 0:
            mean_f1 = np.mean(metrics['f1']) if metrics['f1'] else 0
            std_f1 = np.std(metrics['f1']) if metrics['f1'] else 0
            mean_p = np.mean(metrics['precision']) if metrics['precision'] else 0
            std_p = np.std(metrics['precision']) if metrics['precision'] else 0
            mean_r = np.mean(metrics['recall']) if metrics['recall'] else 0
            std_r = np.std(metrics['recall']) if metrics['recall'] else 0
            
            print(f"{dataset_lang:<15} | {experiment:<15} | {mean_f1:>5.2f} ± {std_f1:<4.2f}% | {mean_p:>5.2f} ± {std_p:<4.2f}% | {mean_r:>5.2f} ± {std_r:<4.2f}% | {seeds_found}")

    print("="*110)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate NER results across seeds')
    parser.add_argument('--dir', type=str, default='/content/drive/My Drive/olfaction_inspired_ner', help='Base results directory')
    args = parser.parse_args()
    
    aggregate_results(args.dir)
