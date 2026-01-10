"""
Analysis tools for receptor activations.
This is critical for demonstrating the biological inspiration works.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict
import os


def analyze_receptor_activations(model, data_loader, vocab_info, device, save_dir='./analysis'):
    """
    Analyze receptor activation patterns.
    
    This generates visualizations that are crucial for the paper:
    1. Receptor activation heatmap per entity type
    2. Top activating tokens per receptor
    3. t-SNE visualization of glomerular representations
    4. Receptor specialization metrics
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    idx2word = vocab_info['idx2word']
    idx2label = vocab_info['idx2label']
    
    # Collect activations
    print("Collecting receptor activations...")
    receptor_activations_by_entity = defaultdict(list)
    glomeruli_activations_by_entity = defaultdict(list)
    token_activations = defaultdict(list)  # receptor -> list of (token, activation)
    
    with torch.no_grad():
        for sentences, tags, lengths in data_loader:
            sentences = sentences.to(device)
            
            # Get activations
            receptors, glomeruli = model.get_receptor_activations(sentences)
            
            if receptors is None:
                print("Model does not have receptors (baseline model?)")
                return None
            
            # Process each sequence
            for i in range(len(sentences)):
                length = lengths[i].item()
                
                for j in range(length):
                    token_idx = sentences[i, j].item()
                    token = idx2word[token_idx]
                    label = idx2label[tags[i, j].item()]
                    
                    # Store by entity type (skip 'O' tags)
                    if label != 'O':
                        entity_type = label.split('-')[1] if '-' in label else label
                        receptor_activations_by_entity[entity_type].append(
                            receptors[i, j].cpu().numpy()
                        )
                        glomeruli_activations_by_entity[entity_type].append(
                            glomeruli[i, j].cpu().numpy()
                        )
                    
                    # Store top activations per receptor
                    receptor_acts = receptors[i, j].cpu().numpy()
                    for receptor_idx, activation in enumerate(receptor_acts):
                        if activation > 0.1:  # Only store significant activations
                            token_activations[receptor_idx].append((token, activation))
    
    results = {}
    
    # 1. Receptor activation heatmap per entity type
    print("Creating receptor activation heatmap...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    entity_types = sorted(receptor_activations_by_entity.keys())
    mean_activations = []
    
    for entity in entity_types:
        acts = np.array(receptor_activations_by_entity[entity])
        mean_act = acts.mean(axis=0)
        mean_activations.append(mean_act)
    
    mean_activations = np.array(mean_activations)
    
    sns.heatmap(mean_activations, 
                xticklabels=range(0, mean_activations.shape[1], 10),
                yticklabels=entity_types,
                cmap='YlOrRd',
                cbar_kws={'label': 'Mean Activation'},
                ax=ax)
    ax.set_xlabel('Receptor Index')
    ax.set_ylabel('Entity Type')
    ax.set_title('Mean Receptor Activations by Entity Type')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'receptor_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    results['mean_activations'] = mean_activations
    results['entity_types'] = entity_types
    
    # 2. Top activating tokens per receptor
    print("Finding top activating tokens per receptor...")
    top_k = 10
    receptor_interpretations = {}
    
    for receptor_idx, token_acts in token_activations.items():
        if len(token_acts) > 0:
            # Sort by activation
            top_tokens = sorted(token_acts, key=lambda x: x[1], reverse=True)[:top_k]
            receptor_interpretations[int(receptor_idx)] = [
                {'token': token, 'activation': float(act)} 
                for token, act in top_tokens
            ]
    
    results['receptor_interpretations'] = receptor_interpretations
    
    # Print some examples
    print("\nTop 5 receptors and their top tokens:")
    for receptor_idx in sorted(receptor_interpretations.keys())[:5]:
        print(f"\nReceptor {receptor_idx}:")
        for item in receptor_interpretations[receptor_idx][:5]:
            print(f"  {item['token']}: {item['activation']:.3f}")
    
    # 3. t-SNE visualization of glomerular representations
    print("\nCreating t-SNE visualization...")
    
    # Sample for t-SNE (use at most 1000 points for speed)
    all_glomeruli = []
    all_labels = []
    
    for entity, acts in glomeruli_activations_by_entity.items():
        sample_size = min(200, len(acts))
        sampled_acts = np.array(acts[:sample_size])
        all_glomeruli.append(sampled_acts)
        all_labels.extend([entity] * sample_size)
    
    all_glomeruli = np.concatenate(all_glomeruli, axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    glomeruli_2d = tsne.fit_transform(all_glomeruli)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(entity_types)))
    for i, entity in enumerate(entity_types):
        indices = [j for j, label in enumerate(all_labels) if label == entity]
        ax.scatter(glomeruli_2d[indices, 0], 
                  glomeruli_2d[indices, 1],
                  c=[colors[i]], 
                  label=entity, 
                  alpha=0.6, 
                  s=50)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Glomerular Representations')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'glomeruli_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Compute receptor specialization metrics
    print("\nComputing receptor specialization metrics...")
    
    # Sparsity: what fraction of receptors are active on average?
    all_receptor_acts = []
    for entity, acts in receptor_activations_by_entity.items():
        all_receptor_acts.extend(acts)
    all_receptor_acts = np.array(all_receptor_acts)
    
    sparsity = (all_receptor_acts > 0.1).mean()
    avg_activation = all_receptor_acts[all_receptor_acts > 0.1].mean() if all_receptor_acts.max() > 0.1 else 0
    
    results['sparsity'] = float(sparsity)
    results['avg_activation'] = float(avg_activation)
    
    print(f"Receptor activation sparsity: {sparsity:.2%}")
    print(f"Average activation (when active): {avg_activation:.3f}")
    
    # Entity-specific activation patterns
    entity_specificity = {}
    for entity in entity_types:
        acts = np.array(receptor_activations_by_entity[entity])
        entity_mean = acts.mean(axis=0)
        
        # Compute variance to find specialized receptors
        variance = entity_mean.var()
        entity_specificity[entity] = float(variance)
    
    results['entity_specificity'] = entity_specificity
    
    # Save results
    import json
    with open(os.path.join(save_dir, 'receptor_analysis.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON
        save_results = {
            'entity_types': results['entity_types'],
            'sparsity': results['sparsity'],
            'avg_activation': results['avg_activation'],
            'entity_specificity': results['entity_specificity'],
            'receptor_interpretations': results['receptor_interpretations']
        }
        json.dump(save_results, f, indent=2)
    
    print(f"\n✓ Analysis complete! Results saved to {save_dir}")
    
    return results


def compare_models(results_dirs, save_dir='./comparison'):
    """
    Compare results from multiple experiments.
    
    Args:
        results_dirs: Dict mapping experiment name to results directory
        save_dir: Where to save comparison visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    import json
    
    # Load all results
    all_results = {}
    for name, results_dir in results_dirs.items():
        results_path = os.path.join(results_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                all_results[name] = json.load(f)
    
    if not all_results:
        print("No results found!")
        return
    
    # Extract test F1 scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall F1 comparison
    names = list(all_results.keys())
    f1_scores = [all_results[name]['test']['f1'] for name in names]
    
    axes[0].bar(range(len(names)), f1_scores, color='skyblue', edgecolor='black')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('Test F1 Score Comparison')
    axes[0].set_ylim([min(f1_scores) - 0.02, max(f1_scores) + 0.02])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Per-entity F1 comparison
    entity_types = list(all_results[names[0]]['test']['per_entity'].keys())
    x = np.arange(len(entity_types))
    width = 0.8 / len(names)
    
    for i, name in enumerate(names):
        per_entity = all_results[name]['test']['per_entity']
        scores = [per_entity.get(entity, 0) for entity in entity_types]
        axes[1].bar(x + i * width, scores, width, label=name, alpha=0.8)
    
    axes[1].set_xlabel('Entity Type')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Per-Entity F1 Score Comparison')
    axes[1].set_xticks(x + width * (len(names) - 1) / 2)
    axes[1].set_xticklabels(entity_types)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison table
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Model':<25} {'Test F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*60)
    
    for name in names:
        test = all_results[name]['test']
        print(f"{name:<25} {test['f1']:<10.4f} {test['precision']:<12.4f} {test['recall']:<10.4f}")
    
    print("="*60)
    
    print(f"\n✓ Comparison saved to {save_dir}")


if __name__ == '__main__':
    print("Receptor analysis module loaded.")
    print("Use analyze_receptor_activations() to analyze a trained model.")
