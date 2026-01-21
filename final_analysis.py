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
    # Structure: BASE_SAVE_DIR / dataset / language / experiment
    
    # helper to find experiments
    results_data = []

    # Get all subdirectories (datasets) in BASE_SAVE_DIR
    try:
        datasets_found = [d for d in os.listdir(BASE_SAVE_DIR) if os.path.isdir(os.path.join(BASE_SAVE_DIR, d))]
    except:
        datasets_found = []

    for dataset in datasets_found:
        dataset_path = os.path.join(BASE_SAVE_DIR, dataset)
        languages = [l for l in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, l))]
        
        for lang in languages:
            lang_path = os.path.join(dataset_path, lang)
            experiments = [e for e in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, e))]
            
            # Store results for this dataset_language combo
            dataset_results = []
            
            for exp in experiments:
                res_file = os.path.join(lang_path, exp, 'results.json')
                if os.path.exists(res_file):
                    try:
                        with open(res_file, 'r') as f:
                            data = json.load(f)
                        
                        test_metrics = data['test']
                        
                        # Extract basic metrics
                        row = {
                            'Experiment': exp,
                            'F1': test_metrics['f1'],
                            'Precision': test_metrics['precision'],
                            'Recall': test_metrics['recall']
                        }
                        
                        # Extract Per-Entity F1s
                        per_entity = test_metrics.get('per_entity', {})
                        for entity, f1 in per_entity.items():
                            row[f"{entity}_F1"] = f1
                        
                        # Calculate/Extract Averages
                        # Assuming the metrics dict might have them, or we verify
                        # 'f1' usually is micro or weighted depending on implementation. 
                        # seqeval classification_report info is usually parsed.
                        # For now, we take what's in JSON. 
                        # If 'weighted' or 'macro' missing, we might not be able to recalc easily without raw predictions.
                        # But we can assume 'avg' scores if available.
                        # Let's map what we have.
                        
                        row['Micro_Avg'] = test_metrics.get('f1') # Usually main F1 is micro in seqeval for NER? Or check implementation.
                        # seqeval default is micro-average F1 score.
                        
                        # If implementation saved detailed report, we could parse it. 
                        # But let's assume standard values.
                        
                        # Add to list
                        dataset_results.append(row)
                    except Exception as e:
                        print(f"Error loading {res_file}: {e}")

            if not dataset_results:
                continue

            # Create DataFrame
            df = pd.DataFrame(dataset_results)
            
            # Sort by F1
            df = df.sort_values(by='F1', ascending=False)
            
            # Display Table
            print(f"\n{'='*100}")
            print(f"RESULTS FOR: {dataset} ({lang})")
            print(f"{'='*100}")
            
            # Reorder columns: Exp, F1, P, R, then Entities...
            cols = ['Experiment', 'F1', 'Precision', 'Recall']
            # Add entity cols sorted
            entity_cols = sorted([c for c in df.columns if c.endswith('_F1')])
            cols.extend(entity_cols)
            
            # Add remaining (like Micro/Macro if we had separate ones)
            # Filter cols that exist
            cols = [c for c in cols if c in df.columns]
            
            print(df[cols].round(4).to_string(index=False))
            print(f"{'='*100}\n")
            
            # ------------------------------------------------------------------
            # Visualization 1: Per-Entity F1 Comparison (Bar Chart)
            # ------------------------------------------------------------------
            if entity_cols:
                # Top 8 models or all? User asked for "Top 8 Models" in title of image given
                # We'll plot all available.
                
                # Reshape for seaborn
                df_long = df.melt(id_vars='Experiment', value_vars=entity_cols, 
                                  var_name='Entity Type', value_name='F1 Score')
                
                # Clean Entity Type name (remove _F1)
                df_long['Entity Type'] = df_long['Entity Type'].str.replace('_F1', '')
                
                plt.figure(figsize=(14, 8))
                sns.barplot(data=df_long, x='Entity Type', y='F1 Score', hue='Experiment', palette='tab10', edgecolor='black')
                plt.title(f'Per-Entity F1 Score Comparison ({dataset}-{lang})', fontsize=16, fontweight='bold')
                plt.ylim(0, 1.05)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()
                
            # ------------------------------------------------------------------
            # Visualization 2: Precision vs Recall Bubble Chart
            # ------------------------------------------------------------------
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot
            # X=Recall, Y=Precision, Size=F1
            sns.scatterplot(data=df, x='Recall', y='Precision', 
                            size='F1', sizes=(200, 1000), 
                            hue='Experiment', palette='Set2', alpha=0.7, edgecolor='black', legend=False)
            
            # Add labels
            for i, row in df.iterrows():
                plt.text(row['Recall']+0.0005, row['Precision'], row['Experiment'], fontsize=9)
            
            # Plot reference line? No, just the points.
            
            plt.title(f'Precision vs Recall (bubble size = F1 score) - {dataset}-{lang}', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.show()

print("\nAnalysis Complete.")
