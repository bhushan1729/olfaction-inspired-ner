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

    for dataset in sorted(os.listdir(base_dir)):
        dataset_dir = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_dir): continue

        for lang in sorted(os.listdir(dataset_dir)):
            lang_dir = os.path.join(dataset_dir, lang)
            if not os.path.isdir(lang_dir): continue

            for exp in sorted(os.listdir(lang_dir)):
                exp_dir = os.path.join(lang_dir, exp)
                if not os.path.isdir(exp_dir): continue

                # Collect metrics across seeds
                f1_scores = []
                precisions = []
                recalls = []
                seeds_found = 0

                for seed_folder in os.listdir(exp_dir):
                    if not seed_folder.startswith('seed_'): continue
                    
                    results_file = os.path.join(exp_dir, seed_folder, 'results.json')
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                                test_metrics = data.get('test', {})
                                
                                if 'f1' in test_metrics:
                                    f1_scores.append(test_metrics['f1'] * 100) # Convert to percentage
                                if 'precision' in test_metrics:
                                    precisions.append(test_metrics['precision'] * 100)
                                if 'recall' in test_metrics:
                                    recalls.append(test_metrics['recall'] * 100)
                                    
                                seeds_found += 1
                        except Exception as e:
                            pass
                
                if seeds_found > 0:
                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    mean_p = np.mean(precisions) if precisions else 0
                    std_p = np.std(precisions) if precisions else 0
                    mean_r = np.mean(recalls) if recalls else 0
                    std_r = np.std(recalls) if recalls else 0
                    
                    dataset_lang = f"{dataset}_{lang}" if lang != 'default' else dataset
                    
                    print(f"{dataset_lang:<15} | {exp:<15} | {mean_f1:>5.2f} ± {std_f1:<4.2f}% | {mean_p:>5.2f} ± {std_p:<4.2f}% | {mean_r:>5.2f} ± {std_r:<4.2f}% | {seeds_found}")

    print("="*110)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate NER results across seeds')
    parser.add_argument('--dir', type=str, default='/content/drive/My Drive/olfaction_inspired_ner', help='Base results directory')
    args = parser.parse_args()
    
    aggregate_results(args.dir)
