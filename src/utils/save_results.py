"""
Save experiment results in standardized format for GitHub tracking.
Automatically saves metrics, config, and visualizations.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def save_experiment_results(
    experiment_name,
    dataset_name,
    model_type,
    config,
    results,
    visualization_dir=None,
    output_dir='./experiment_results'
):
    """
    Save experiment results in standardized format.
    
    Args:
        experiment_name: Name of experiment (e.g., 'baseline_conll2003')
        dataset_name: Dataset used (e.g., 'CoNLL-2003', 'OntoNotes5')
        model_type: Model architecture (e.g., 'baseline', 'olfactory_gelu')
        config: Full experiment configuration dict
        results: Results dict with train/valid/test metrics
        visualization_dir: Directory containing visualizations
        output_dir: Where to save results
    """
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / dataset_name / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save metadata
    metadata = {
        'experiment_name': experiment_name,
        'dataset': dataset_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'config': config,
    }
    
    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 2. Save results
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 3. Save summary (human-readable)
    summary_lines = [
        f"# {experiment_name}",
        f"",
        f"**Dataset**: {dataset_name}",
        f"**Model**: {model_type}",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Configuration",
        f"",
        f"```yaml",
    ]
    
    # Add key config
    for key in ['embed_dim', 'num_receptors', 'num_glomeruli', 'receptor_activation', 
                'lstm_hidden', 'dropout', 'learning_rate', 'batch_size']:
        if key in config:
            summary_lines.append(f"{key}: {config[key]}")
    
    summary_lines.extend([
        f"```",
        f"",
        f"## Results",
        f"",
        f"### Test Performance",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| **F1** | **{results['test']['f1']:.4f}** |",
        f"| Precision | {results['test']['precision']:.4f} |",
        f"| Recall | {results['test']['recall']:.4f} |",
        f"",
    ])
    
    # Per-entity scores
    if 'per_entity' in results['test']:
        summary_lines.extend([
            f"### Per-Entity F1 Scores",
            f"",
            f"| Entity | F1 |",
            f"|--------|-----|",
        ])
        
        for entity, f1 in sorted(results['test']['per_entity'].items()):
            if entity not in ['micro avg', 'macro avg', 'weighted avg']:
                summary_lines.append(f"| {entity} | {f1:.4f} |")
        
        summary_lines.append("")
    
    # Training info
    if 'epochs' in results and len(results['epochs']) > 0:
        last_epoch = results['epochs'][-1]
        summary_lines.extend([
            f"### Training",
            f"",
            f"- Epochs trained: {len(results['epochs'])}",
            f"- Best validation F1: {results.get('best_f1', 'N/A')}",
            f""
        ])
    
    with open(exp_dir / 'SUMMARY.md', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # 4. Copy visualizations
    if visualization_dir and os.path.exists(visualization_dir):
        vis_dest = exp_dir / 'visualizations'
        vis_dest.mkdir(exist_ok=True)
        
        for file in Path(visualization_dir).glob('*.png'):
            shutil.copy(file, vis_dest / file.name)
        
        print(f"✓ Copied visualizations to {vis_dest}")
    
    print(f"\n✓ Results saved to {exp_dir}")
    print(f"  - metadata.json: Experiment configuration")
    print(f"  - results.json: Full results")
    print(f"  - SUMMARY.md: Human-readable summary")
    
    return exp_dir


def generate_results_index(output_dir='./experiment_results'):
    """Generate index of all experiments for GitHub README."""
    
    results_path = Path(output_dir)
    if not results_path.exists():
        print("No results directory found")
        return
    
    # Collect all experiments
    experiments = []
    
    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        
        for exp_dir in dataset_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Load metadata and results
            metadata_file = exp_dir / 'metadata.json'
            results_file = exp_dir / 'results.json'
            
            if metadata_file.exists() and results_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                with open(results_file) as f:
                    results = json.load(f)
                
                experiments.append({
                    'dataset': dataset_name,
                    'name': exp_dir.name,
                    'model': metadata.get('model_type', 'unknown'),
                    'activation': metadata.get('config', {}).get('receptor_activation', 'N/A'),
                    'f1': results.get('test', {}).get('f1', 0.0),
                    'precision': results.get('test', {}).get('precision', 0.0),
                    'recall': results.get('test', {}).get('recall', 0.0),
                    'path': str(exp_dir.relative_to(results_path))
                })
    
    # Sort by dataset, then F1
    experiments.sort(key=lambda x: (x['dataset'], -x['f1']))
    
    # Generate markdown
    lines = [
        "# Experiment Results",
        "",
        "Comprehensive tracking of all NER experiments across datasets and model variants.",
        "",
        "**Last updated**: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "",
    ]
    
    # Group by dataset
    for dataset in sorted(set(e['dataset'] for e in experiments)):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Experiment | Model | Activation | F1 | Precision | Recall | Details |")
        lines.append("|------------|-------|------------|-----|-----------|--------|---------|")
        
        dataset_exps = [e for e in experiments if e['dataset'] == dataset]
        for exp in dataset_exps:
            details_link = f"[view]({exp['path']}/SUMMARY.md)"
            lines.append(
                f"| {exp['name']} | {exp['model']} | {exp['activation']} | "
                f"**{exp['f1']:.4f}** | {exp['precision']:.4f} | {exp['recall']:.4f} | {details_link} |"
            )
        
        lines.append("")
    
    # Write index
    with open(results_path / 'README.md', 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ Generated results index: {results_path / 'README.md'}")


if __name__ == '__main__':
    # Example usage
    print("Example: save_experiment_results(...)")
    print("\nTo generate index: generate_results_index()")
