#!/usr/bin/env python3
"""
Generate heatmaps for receptor and glomeruli layer activations.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.dataset import prepare_data
from src.model.olfactory_ner import create_olfactory_ner
from src.train import load_config


def get_layer_activations(model, data_loader, device, layer_name='receptor'):
    """
    Extract activations from receptor or glomeruli layer.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run on
        layer_name: 'receptor' or 'glomeruli'
    
    Returns:
        activations: Dict mapping entity types to activation matrices
    """
    model.eval()
    
    # Storage for activations per entity type
    entity_activations = {
        'PER': [],
        'LOC': [],
        'ORG': [],
        'MISC': [],
        'O': []
    }
    
    with torch.no_grad():
        for sentences, tags, lengths in data_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            
            # Get embeddings
            embedded = model.embedding(sentences)
            
            # Get activations based on layer
            if layer_name == 'receptor' and hasattr(model, 'receptor_layer'):
                # Pass through receptor layer
                activations = model.receptor_layer(embedded)  # [batch, seq_len, num_receptors]
            elif layer_name == 'glomeruli' and hasattr(model, 'glomerular_layer'):
                # Pass through receptor and glomeruli layers
                receptor_out = model.receptor_layer(embedded)
                activations = model.glomerular_layer(receptor_out)  # [batch, seq_len, num_glomeruli]
            else:
                continue
            
            # Process each sequence in batch
            for i, length in enumerate(lengths):
                seq_activations = activations[i, :length].cpu().numpy()  # [seq_len, num_features]
                seq_tags = tags[i, :length].cpu().numpy()
                
                # Group activations by entity type
                for t in range(length):
                    tag_idx = seq_tags[t]
                    
                    # Map tag index to entity type
                    if tag_idx == 0:  # B-PER or I-PER (depends on label mapping)
                        entity_type = 'PER'
                    elif tag_idx == 1:  # B-LOC or I-LOC
                        entity_type = 'LOC'
                    elif tag_idx == 2:  # B-ORG or I-ORG
                        entity_type = 'ORG'
                    elif tag_idx == 3:  # B-MISC or I-MISC
                        entity_type = 'MISC'
                    else:
                        entity_type = 'O'
                    
                    entity_activations[entity_type].append(seq_activations[t])
    
    # Compute mean activations per entity type
    mean_activations = {}
    for entity_type, acts in entity_activations.items():
        if len(acts) > 0:
            mean_activations[entity_type] = np.mean(acts, axis=0)
    
    return mean_activations


def plot_heatmap(activations, entity_types, save_path, title, layer_name):
    """
    Plot heatmap of activations.
    
    Args:
        activations: Dict mapping entity types to activation vectors
        entity_types: List of entity types
        save_path: Path to save the plot
        title: Plot title
        layer_name: 'Receptor' or 'Glomeruli'
    """
    # Create activation matrix
    activation_matrix = []
    valid_entities = []
    
    for entity in entity_types:
        if entity in activations:
            activation_matrix.append(activations[entity])
            valid_entities.append(entity)
    
    if not activation_matrix:
        print(f"⚠️  No activations found for {title}")
        return
    
    activation_matrix = np.array(activation_matrix)  # [num_entities, num_features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot heatmap
    sns.heatmap(
        activation_matrix,
        yticklabels=valid_entities,
        xticklabels=[f'{layer_name[0]}{i+1}' for i in range(activation_matrix.shape[1])],
        cmap='YlOrRd',
        cbar_kws={'label': 'Mean Activation'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'{layer_name} Units', fontsize=12)
    ax.set_ylabel('Entity Type', fontsize=12)
    
    # Adjust x-axis labels to show every 10th unit for readability
    if activation_matrix.shape[1] > 50:
        xticks = ax.get_xticks()
        xticklabels = [f'{layer_name[0]}{i+1}' if (i+1) % 10 == 0 else '' 
                      for i in range(activation_matrix.shape[1])]
        ax.set_xticklabels(xticklabels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap: {save_path}")


def generate_heatmaps_for_experiment(experiment_name, model_path, config, vocab_info, 
                                     test_loader, device, output_dir):
    """Generate heatmaps for a single experiment."""
    
    # Check if model is olfactory type
    if config.get('model_type') != 'olfactory':
        print(f"⚠️  Skipping {experiment_name} (not an olfactory model)")
        return
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        return
    
    print(f"\nGenerating heatmaps for: {experiment_name}")
    
    # Load model
    vocab_size = len(vocab_info['word2idx'])
    num_tags = len(vocab_info['label2idx'])
    
    model = create_olfactory_ner(vocab_size, num_tags, config)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Entity types
    entity_types = ['PER', 'LOC', 'ORG', 'MISC', 'O']
    
    # Generate receptor heatmap
    if hasattr(model, 'receptor_layer'):
        print("  Generating receptor heatmap...")
        receptor_activations = get_layer_activations(model, test_loader, device, 'receptor')
        
        receptor_heatmap_path = os.path.join(output_dir, f'receptor_heatmap_{experiment_name}.png')
        plot_heatmap(
            receptor_activations,
            entity_types,
            receptor_heatmap_path,
            f'Receptor Layer Activations - {experiment_name}',
            'Receptor'
        )
    
    # Generate glomeruli heatmap
    if hasattr(model, 'glomerular_layer') and config.get('use_glomeruli', True):
        print("  Generating glomeruli heatmap...")
        glomeruli_activations = get_layer_activations(model, test_loader, device, 'glomeruli')
        
        glomeruli_heatmap_path = os.path.join(output_dir, f'glomeruli_heatmap_{experiment_name}.png')
        plot_heatmap(
            glomeruli_activations,
            entity_types,
            glomeruli_heatmap_path,
            f'Glomeruli Layer Activations - {experiment_name}',
            'Glomeruli'
        )


def main():
    """Main execution function."""
    print("=" * 80)
    print("HEATMAP GENERATION FOR OLFACTION-INSPIRED NER")
    print("=" * 80)
    
    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data (we'll use the same data for all experiments)
    print("\nLoading CoNLL-2003 dataset...")
    device = torch.device('cpu')
    
    train_loader, valid_loader, test_loader, vocab_info = prepare_data(
        data_dir='./data/raw',
        batch_size=32,
        min_freq=2
    )
    
    print(f"✓ Data loaded (Test set: {len(test_loader.dataset)} samples)")
    
    # Find all experiment results
    experiments_to_process = []
    
    # Check results directory
    results_dir = Path('results')
    if results_dir.exists():
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                model_path = exp_dir / 'best_model.pt'
                results_path = exp_dir / 'results.json'
                
                if model_path.exists() and results_path.exists():
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    
                    config = results.get('config', {})
                    experiments_to_process.append({
                        'name': exp_dir.name,
                        'model_path': str(model_path),
                        'config': config
                    })
    
    print(f"\nFound {len(experiments_to_process)} experiments with trained models:")
    for exp in experiments_to_process:
        print(f"  - {exp['name']}")
    
    # Generate heatmaps for each experiment
    print("\n" + "=" * 80)
    print("GENERATING HEATMAPS")
    print("=" * 80)
    
    for exp in experiments_to_process:
        generate_heatmaps_for_experiment(
            exp['name'],
            exp['model_path'],
            exp['config'],
            vocab_info,
            test_loader,
            device,
            output_dir
        )
    
    print("\n" + "=" * 80)
    print("HEATMAP GENERATION COMPLETED!")
    print("=" * 80)
    print(f"All heatmaps saved to: {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
