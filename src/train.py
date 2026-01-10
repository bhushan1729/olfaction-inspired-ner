"""
Training script for NER models.
Supports both baseline and olfactory-inspired models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os
import yaml
import argparse
from tqdm import tqdm
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import prepare_data, load_glove_embeddings
from src.model.baseline import create_baseline_ner
from src.model.olfactory_ner import create_olfactory_ner
from src.training.evaluate import evaluate_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path, experiment_name):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get experiment-specific config
    exp_config = config[experiment_name]
    
    # Merge with data and training configs
    exp_config.update(config.get('data', {}))
    exp_config.update(config.get('training', {}))
    
    return exp_config


def train_epoch(model, train_loader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_sparse_loss = 0
    total_diverse_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (sentences, tags, lengths) in enumerate(pbar):
        sentences = sentences.to(device)
        tags = tags.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        task_loss = model(sentences, tags, lengths)
        
        # Add regularization losses
        sparse_loss = torch.tensor(0.0, device=device)
        diverse_loss = model.get_diversity_loss()
        
        # Total loss
        loss = task_loss + \
               config.get('lambda_sparse', 0.0) * sparse_loss + \
               config.get('lambda_diverse', 0.0) * diverse_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 5.0))
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_sparse_loss += sparse_loss.item()
        total_diverse_loss += diverse_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'task': f'{task_loss.item():.4f}'
        })
    
    n_batches = len(train_loader)
    return {
        'total_loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches,
        'sparse_loss': total_sparse_loss / n_batches,
        'diverse_loss': total_diverse_loss / n_batches
    }


def train(config, experiment_name, save_dir):
    """Main training function."""
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', 'cuda') == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    train_loader, valid_loader, test_loader, vocab_info = prepare_data(
        data_dir=config.get('data_dir', './data/raw'),
        batch_size=config.get('batch_size', 32),
        min_freq=config.get('min_word_freq', 2)
    )
    
    vocab_size = len(vocab_info['word2idx'])
    num_tags = len(vocab_info['label2idx'])
    
    # Load pre-trained embeddings
    pretrained_embeddings = None
    if config.get('use_pretrained_embeddings', True):
        glove_path = config.get('glove_path', './data/glove.6B.300d.txt')
        if os.path.exists(glove_path):
            print(f"\nLoading GloVe embeddings from {glove_path}...")
            pretrained_embeddings = load_glove_embeddings(
                glove_path, 
                vocab_info['word2idx'], 
                embed_dim=config.get('embed_dim', 300)
            )
        else:
            print(f"\nWarning: GloVe file not found at {glove_path}")
            print("Using random embeddings. Download GloVe from: https://nlp.stanford.edu/projects/glove/")
    
    # Create model
    print("\n" + "="*50)
    print(f"Creating {config['model_type']} model...")
    print("="*50)
    
    if config['model_type'] == 'baseline':
        model = create_baseline_ner(vocab_size, num_tags, config, pretrained_embeddings)
    elif config['model_type'] == 'olfactory':
        model = create_olfactory_ner(vocab_size, num_tags, config, pretrained_embeddings)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(config.get('tensorboard_dir', './runs'), experiment_name))
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_f1 = 0.0
    patience_counter = 0
    max_patience = config.get('early_stopping_patience', 5)
    
    results = {
        'config': config,
        'vocab_size': vocab_size,
        'num_tags': num_tags,
        'label2idx': vocab_info['label2idx'],
        'idx2label': vocab_info['idx2label'],
        'epochs': []
    }
    
    for epoch in range(config.get('max_epochs', 50)):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.get('max_epochs', 50)}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        valid_metrics = evaluate_model(model, valid_loader, vocab_info['idx2label'], device)
        
        # Log metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in valid_metrics.items():
            writer.add_scalar(f'valid/{key}', value, epoch)
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"Valid F1: {valid_metrics['f1']:.4f}")
        print(f"Valid Precision: {valid_metrics['precision']:.4f}")
        print(f"Valid Recall: {valid_metrics['recall']:.4f}")
        
        # Store results
        results['epochs'].append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'valid': valid_metrics
        })
        
        # Scheduler step
        scheduler.step(valid_metrics['f1'])
        
        # Save best model
        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config,
                'vocab_info': vocab_info
            }
            
            os.makedirs(save_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f"\n✓ Saved new best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{max_patience})")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Test evaluation
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, vocab_info['idx2label'], device)
    
    print(f"\nTest F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Per-entity results
    if 'per_entity' in test_metrics:
        print("\nPer-entity F1 scores:")
        for entity, f1 in test_metrics['per_entity'].items():
            print(f"  {entity}: {f1:.4f}")
    
    results['test'] = test_metrics
    results['best_f1'] = best_f1
    
    # Save results
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()
    
    print(f"\n✓ Training complete! Results saved to {save_dir}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NER model')
    parser.add_argument('--config', type=str, default='config/experiments.yaml',
                        help='Path to config file')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., baseline, olfactory_full)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: results/<experiment>)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args.experiment)
    
    # Set save directory
    save_dir = args.save_dir if args.save_dir else os.path.join('results', args.experiment)
    
    # Train
    train(config, args.experiment, save_dir)
