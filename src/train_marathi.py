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

from models.olfactory_ner import OlfactoryNERModel
from data.dataset_marathi import prepare_marathi_data, load_glove_embeddings
from utils.metrics import compute_f1, compute_detailed_metrics


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_task_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for sentences, labels, lengths in pbar:
        sentences = sentences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sentences, lengths)
        
        # Reshape for loss computation
        outputs_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)
        
        # Compute loss
        task_loss = criterion(outputs_flat, labels_flat)
        
        # Backward pass
        task_loss.backward()
        optimizer.step()
        
        total_loss += task_loss.item()
        total_task_loss += task_loss.item()
        
        pbar.set_postfix(loss=task_loss.item(), task=task_loss.item())
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device, idx2label):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        for sentences, labels, lengths in pbar:
            sentences = sentences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            outputs = model(sentences, lengths)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=-1)
            
            # Convert to lists for metric computation
            for i in range(len(sentences)):
                length = lengths[i].item()
                pred_seq = predictions[i, :length].cpu().tolist()
                label_seq = labels[i, :length].cpu().tolist()
                
                # Convert to label strings
                pred_labels = [idx2label[p] for p in pred_seq]
                true_labels = [idx2label[l] for l in label_seq]
                
                all_predictions.append(pred_labels)
                all_labels.append(true_labels)
    
    # Compute metrics
    f1, precision, recall = compute_f1(all_predictions, all_labels)
    
    return f1, precision, recall, all_predictions, all_labels


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
    
    word2idx = vocab_info['word2idx']
    label2idx = vocab_info['label2idx']
    idx2label = vocab_info['idx2label']
    
    # Load embeddings
    embedding_matrix = load_glove_embeddings(
        config.get('glove_path', './data/glove.6B.300d.txt'),
        word2idx,
        config.get('embedding_dim', 300)
    )
    
    # Create model
    print("\n" + "=" * 50)
    print("Creating olfactory model...")
    print("=" * 50)
    model = OlfactoryNERModel(
        vocab_size=len(word2idx),
        embedding_dim=config['embedding_dim'],
        num_receptors=config['num_receptors'],
        num_glomeruli=config['num_glomeruli'],
        num_labels=len(label2idx),
        dropout=config.get('dropout', 0.2),
        activation=config.get('activation', 'relu'),
        pretrained_embeddings=torch.FloatTensor(embedding_matrix)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_f1 = 0
    patience = config.get('patience', 5)
    patience_counter = 0
    
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print("=" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        print("\nEvaluating on validation set...")
        valid_f1, valid_precision, valid_recall, _, _ = evaluate(
            model, valid_loader, criterion, device, idx2label
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Valid F1: {valid_f1:.4f}")
        print(f"Valid Precision: {valid_precision:.4f}")
        print(f"Valid Recall: {valid_recall:.4f}")
        
        # Save best model
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            patience_counter = 0
            
            # Save model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"\n✓ Saved new best model (F1: {valid_f1:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print("\nEarly stopping!")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")
    print("=" * 50)
    test_f1, test_precision, test_recall, test_predictions, test_labels = evaluate(
        model, test_loader, criterion, device, idx2label
    )
    
    print(f"\nTest F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Compute detailed metrics
    detailed_metrics = compute_detailed_metrics(test_predictions, test_labels)
    
    print("\nPer-entity F1 scores:")
    for entity, score in detailed_metrics.items():
        print(f"  {entity}: {score:.4f}")
    
    # Save results
    results = {
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'best_valid_f1': best_f1,
        'detailed_metrics': detailed_metrics,
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
