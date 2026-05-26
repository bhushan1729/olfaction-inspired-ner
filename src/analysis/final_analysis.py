# @title 4. Comprehensive Analysis & Visualizations
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Use the BASE_SAVE_DIR from setup
# If running fresh, ensure these are set:
if 'BASE_SAVE_DIR' not in locals():
    BASE_SAVE_DIR = '/content/drive/My Drive/olfaction_inspired_ner' if os.path.exists('/content/drive') else './results'

CONFIG_PATH = 'config/universal_config.yaml'
import yaml
with open(CONFIG_PATH, 'r') as f:
    full_config = yaml.safe_load(f)

# Define which datasets to look for (or scan directory)
# We'll scan the directory to be comprehensive
if not os.path.exists(BASE_SAVE_DIR):
    print(f"Directory {BASE_SAVE_DIR} does not exist.")
else:
    from collections import defaultdict
    
    # helper to find and aggregate experiments across seeds
    known_experiments = {
        'baseline', 'olfactory', 'receptors_only', 'no_sparsity', 
        'more_receptors', 'more_glomeruli', 'more_receptors_more_glomeruli'
    }
    known_datasets = {
        'conll_en', 'conll_en_1k', 'wikiann_mr', 'wikiann_mr_1k', 
        'wikiann_hi', 'wikiann_hi_1k', 'wikiann_ta', 'wikiann_ta_1k', 
        'wikiann_bn', 'wikiann_bn_1k', 'wikiann_te', 'wikiann_te_1k'
    }
    
    raw_metrics = defaultdict(lambda: {
        'f1': [],
        'precision': [],
        'recall': [],
        'entities': defaultdict(list)
    })
    
    for root, dirs, files in os.walk(BASE_SAVE_DIR):
        if 'results.json' in files:
            results_file = os.path.join(root, 'results.json')
            rel_path = os.path.relpath(root, BASE_SAVE_DIR)
            parts = [p for p in rel_path.replace('\\', '/').split('/') if p and p != '.']
            
            dataset_lang = "unknown_dataset"
            experiment = "unknown_experiment"
            
            remaining_parts = []
            for part in parts:
                part_lower = part.lower()
                if part_lower.startswith('seed_') or part_lower.isdigit():
                    continue
                elif part_lower in known_experiments:
                    experiment = part
                elif part_lower in known_datasets:
                    dataset_lang = part
                else:
                    remaining_parts.append(part)
            
            if dataset_lang == "unknown_dataset" and len(parts) >= 1:
                dataset_lang = parts[0]
            if experiment == "unknown_experiment":
                if len(parts) == 3:
                    experiment = parts[1]
                elif len(parts) == 4:
                    experiment = parts[2]
                elif remaining_parts:
                    experiment = remaining_parts[-1]
            
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                test_metrics = data.get('test', {})
                f1 = test_metrics.get('f1')
                precision = test_metrics.get('precision')
                recall = test_metrics.get('recall')
                
                if f1 is not None:
                    raw_metrics[(dataset_lang, experiment)]['f1'].append(f1)
                if precision is not None:
                    raw_metrics[(dataset_lang, experiment)]['precision'].append(precision)
                if recall is not None:
                    raw_metrics[(dataset_lang, experiment)]['recall'].append(recall)
                
                per_entity = test_metrics.get('per_entity', {})
                for entity, score in per_entity.items():
                    raw_metrics[(dataset_lang, experiment)]['entities'][entity].append(score)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                
    # Now group by dataset to mimic original script flow
    datasets_data = defaultdict(list)
    for (dataset_lang, experiment), metrics in raw_metrics.items():
        if not metrics['f1']:
            continue
        row = {
            'Experiment': experiment,
            'F1': np.mean(metrics['f1']),
            'Precision': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
            'Recall': np.mean(metrics['recall']) if metrics['recall'] else 0.0
        }
        for entity, scores in metrics['entities'].items():
            row[f"{entity}_F1"] = np.mean(scores)
        datasets_data[dataset_lang].append(row)
        
    for dataset, dataset_results in datasets_data.items():
        df = pd.DataFrame(dataset_results)
        df = df.sort_values(by='F1', ascending=False)
        
        # Display Table
        print(f"\n{'='*100}")
        print(f"RESULTS FOR: {dataset}")
        print(f"{'='*100}")
        
        cols = ['Experiment', 'F1', 'Precision', 'Recall']
        entity_cols = sorted([c for c in df.columns if c.endswith('_F1')])
        cols.extend(entity_cols)
        cols = [c for c in cols if c in df.columns]
        
        print(df[cols].round(4).to_string(index=False))
        print(f"{'='*100}\n")
        
        # ------------------------------------------------------------------
        # Visualization 1: Per-Entity F1 Comparison (Bar Chart)
        # ------------------------------------------------------------------
        if entity_cols:
            df_long = df.melt(id_vars='Experiment', value_vars=entity_cols, 
                              var_name='Entity Type', value_name='F1 Score')
            
            df_long['Entity Type'] = df_long['Entity Type'].str.replace('_F1', '')
            
            plt.figure(figsize=(14, 8))
            sns.barplot(data=df_long, x='Entity Type', y='F1 Score', hue='Experiment', palette='tab10', edgecolor='black')
            plt.title(f'Per-Entity F1 Score Comparison ({dataset})', fontsize=16, fontweight='bold')
            plt.ylim(0, 1.05)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        # ------------------------------------------------------------------
        # Visualization 2: Precision vs Recall Bubble Chart
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 8))
        
        sns.scatterplot(data=df, x='Recall', y='Precision', 
                        size='F1', sizes=(200, 1000), 
                        hue='Experiment', palette='Set2', alpha=0.7, edgecolor='black', legend=False)
        
        for i, row in df.iterrows():
            plt.text(row['Recall']+0.0005, row['Precision'], row['Experiment'], fontsize=9)
        
        plt.title(f'Precision vs Recall (bubble size = F1 score) - {dataset}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

print("\nAnalysis Complete.")
