import os
import subprocess
import json
import numpy as np

seeds = [42, 43, 44, 45, 46]
experiments = ['dense_bottleneck', 'simple_sparse_bottleneck', 'sparse_bottleneck_l1']
dataset_key = 'wikiann_te_1k'

print("Starting control baseline training...")

results = {}

for exp in experiments:
    results[exp] = []
    print(f"\n==================================================")
    print(f"Running experiment: {exp}")
    print(f"==================================================")
    for seed in seeds:
        print(f"--> Training seed {seed}...")
        cmd = [
            'python', 'src/train_universal.py',
            '--dataset_key', dataset_key,
            '--experiment', exp,
            '--seed', str(seed)
        ]
        # Run training
        try:
            # Set UTF-8 encoding environment variable to prevent Windows cp1252 crash
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if res.returncode != 0:
                print(f"Error training seed {seed} for {exp}:")
                print(res.stderr)
        except Exception as e:
            print(f"Exception training seed {seed} for {exp}: {e}")
        
        # Read F1 score
        # The path should be results/wikiann_te_1k/{exp}/seed_{seed}/results.json
        results_path = os.path.join('results', dataset_key, exp, f"seed_{seed}", "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                    f1 = data['test']['f1']
                    # Keep F1 as percentage (e.g., 55.4) or ratio (e.g., 0.554)
                    results[exp].append(f1)
                    print(f"Seed {seed} completed. Test F1: {f1:.4f}")
            except Exception as e:
                print(f"Error reading results from {results_path}: {e}")
        else:
            print(f"Results file not found: {results_path}")

print("\n\n==================================================")
print("Experiment Results Summary")
print("==================================================")
for exp in experiments:
    f1_list = results[exp]
    if len(f1_list) > 0:
        f1_pct = [f * 100 if f < 1.0 else f for f in f1_list]
        mean_f1 = np.mean(f1_pct)
        std_f1 = np.std(f1_pct)
        print(f"{exp}: {mean_f1:.2f}% ± {std_f1:.2f}% (seeds: {f1_pct})")
    else:
        print(f"{exp}: No results collected.")
