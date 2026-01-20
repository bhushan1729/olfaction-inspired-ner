"""
Training script for olfaction-inspired NER model on Marathi dataset.
Based on train.py but adapted for ai4bharat/naamapadam dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml
import os
import json
from tqdm import tqdm
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_marathi import prepare_marathi_data, load_glove_embeddings
from src.model.olfactory_ner import create_olfactory_ner
from src.training.evaluate import evaluate_model


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer,device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_task_loss = 0
    
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'task': f'{task_loss.item():.4f}'
        })
    
    n_batches = len(train_loader)
    return {
        'total_loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches
    }


def train(config, args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("=" * 50)
    print("Loading data...")
    print("=" * 50)
    train_loader, valid_loader, test_loader, vocab_info = prepare_marathi_data(
        cache_dir=config.get('data_dir', './data/naamapadam'),
        batch_size=config['batch_size'],
        min_freq=config.get('min_freq', 2)
    )
    
    vocab_size = len(vocab_info['word2idx'])
    num_tags = len(vocab_info['label2idx'])
    
    # Load embeddings
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
    print("\n" + "=" * 50)
    print("Creating olfactory model...")
    print("=" * 50)
    
    model = create_olfactory_ner(vocab_size, num_tags, config, pretrained_embeddings)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    best_f1 = 0
    patience = config.get('patience', 5)
    patience_counter = 0
    
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    for epoch in range(config.get('num_epochs', 20)):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config.get('num_epochs', 20)}")
        print("=" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        valid_metrics = evaluate_model(model, valid_loader, vocab_info['idx2label'], device)
        
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"Valid F1: {valid_metrics['f1']:.4f}")
        print(f"Valid Precision: {valid_metrics['precision']:.4f}")
        print(f"Valid Recall: {valid_metrics['recall']:.4f}")
        
        # Scheduler step
        scheduler.step(valid_metrics['f1'])
        
        # Save best model
        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            patience_counter = 0
            
            # Save model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config,
                'vocab_info': vocab_info
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"\n✓ Saved new best model (F1: {valid_metrics['f1']:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print("\nEarly stopping!")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")
    print("=" * 50)
    test_metrics = evaluate_model(model, test_loader, vocab_info['idx2label'], device)
    
    print(f"\nTest F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Per-entity results
    if 'per_entity' in test_metrics:
        print("\nPer-entity F1 scores:")
        for entity, f1 in test_metrics['per_entity'].items():
            print(f"  {entity}: {f1:.4f}")
    
    # Save results
    results = {
        'test': test_metrics,
        'best_f1': best_f1,
        'config': config
    }
    
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to {args.save_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train olfactory NER model on Marathi dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if args.experiment not in all_configs:
        raise ValueError(f"Experiment '{args.experiment}' not found in config file")
    
    config = all_configs[args.experiment]
    
    # Train
    results = train(config, args)


if __name__ == '__main__':
    main()
