import argparse
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.bert_loader import get_bert_dataset
from src.model.bert_models import BertBaseline, BertOlfactory

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading {args.dataset} ({args.language})...")
    train_loader, valid_loader, test_loader, vocab_info = get_bert_dataset(
        args.dataset, args.language, batch_size=args.batch_size
    )
    label2idx = vocab_info['label2idx']
    idx2label = vocab_info['idx2label']
    num_labels = len(label2idx)

    # 2. Init Model
    if args.experiment == 'baseline':
        model = BertBaseline(num_labels).to(device)
    elif args.experiment == 'olfactory':
        # Config for olfactory layers
        config = {
            'num_receptors': 128,
            'num_glomeruli': 32,
            'lstm_hidden': 128,
            'lambda_sparse': 0.001
        }
        model = BertOlfactory(num_labels, config).to(device)
    else:
        raise ValueError("Invalid experiment type")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    best_f1 = 0
    save_path = os.path.join(args.save_dir, args.dataset, args.language if args.language else 'en', f"mbert_{args.experiment}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save initial model as fallback
    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
    print("Saved initial model")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            # Forward
            if args.experiment == 'baseline':
                _, loss = model(input_ids, mask, labels)
            else:
                _, loss = model(input_ids, mask, labels) # BertOlfactory returns (None, loss) in training
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Avg Loss: {avg_loss:.4f}")

        # Validation
        f1 = evaluate(model, valid_loader, idx2label, device, args.experiment)
        print(f"Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            print("Saved Best Model")

    # 4. Final Test
    print("Testing Best Model...")
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pt')))
    
    # Get comprehensive test metrics
    test_f1 = evaluate(model, test_loader, idx2label, device, args.experiment, verbose=True)
    
    # Also compute and save full metrics
    from src.training.metrics import compute_ner_metrics
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if args.experiment == 'baseline':
                logits, _ = model(input_ids, mask)
                preds = torch.argmax(logits, dim=2)
                batch_preds = preds.cpu().numpy()
            else:
                preds_list, _ = model(input_ids, mask)
                batch_preds = preds_list
            
            batch_labels = labels.cpu().numpy()
            
            for i in range(len(batch_labels)):
                seq_labels = []
                seq_preds = []
                for j, label_id in enumerate(batch_labels[i]):
                    if label_id != -100:
                        seq_labels.append(idx2label[label_id])
                        if args.experiment == 'baseline':
                            pred_id = batch_preds[i][j]
                        else:
                            pred_id = batch_preds[i][j] if j < len(batch_preds[i]) else 0
                        seq_preds.append(idx2label.get(pred_id, 'O'))
                true_labels.append(seq_labels)
                pred_labels.append(seq_preds)
    
    full_metrics = compute_ner_metrics(true_labels, pred_labels, verbose=False)
    
    # Helper function to convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Save Results
    results = {
        'test_f1': float(test_f1),
        'test_precision': float(full_metrics['precision']),
        'test_recall': float(full_metrics['recall']),
        'test_accuracy': float(full_metrics['accuracy']),
        'best_val_f1': float(best_f1),
        'per_entity_metrics': convert_to_serializable(full_metrics['per_entity']),
        'args': vars(args)
    }
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def evaluate(model, loader, idx2label, device, experiment_type, verbose=False):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if experiment_type == 'baseline':
                logits, _ = model(input_ids, mask)
                preds = torch.argmax(logits, dim=2) # (Batch, Seq)
                batch_preds = preds.cpu().numpy()
            else:
                # BertOlfactory returns list of paths (decoded tags from CRF)
                preds_list, _ = model(input_ids, mask) 
                # preds_list is a list of lists of tag indices
                batch_preds = preds_list

            # Align Predictions with labels
            # We need to reconstruct word_ids to only count first subwords
            batch_labels = labels.cpu().numpy()
            batch_input_ids = input_ids.cpu().numpy()

            for i in range(len(batch_labels)):
                seq_labels = []
                seq_preds = []
                
                # Track which word we're on to only count first subword
                previous_word_idx = None
                
                # We need to reconstruct word boundaries
                # For now, use a simpler approach: count non-padding, non-special tokens
                for j in range(len(batch_labels[i])):
                    label_id = batch_labels[i][j]
                    
                    # Skip padding (label -100 in our new setup means padding only)
                    if label_id == -100:
                        continue
                    
                    # For simplicity in evaluation: include ALL labeled tokens
                    # The CRF learns on all subwords, but we can evaluate on all too
                    # This is actually more informative for subword-level NER
                    seq_labels.append(idx2label[label_id])
                    
                    # Get prediction
                    if experiment_type == 'baseline':
                        pred_id = batch_preds[i][j]
                    else:
                        # For CRF: preds_list[i] is a list of predicted tags for sequence i
                        if j < len(batch_preds[i]):
                            pred_id = batch_preds[i][j]
                        else:
                            pred_id = 0 # Default to 'O'
                    
                    seq_preds.append(idx2label.get(pred_id, 'O'))
                
                if seq_labels:  # Only add if non-empty
                    true_labels.append(seq_labels)
                    pred_labels.append(seq_preds)

    # Compute comprehensive metrics
    from src.training.metrics import compute_ner_metrics
    metrics = compute_ner_metrics(true_labels, pred_labels, verbose=verbose)
    
    return metrics['f1']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--experiment', type=str, choices=['baseline', 'olfactory'], required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    train(args)
