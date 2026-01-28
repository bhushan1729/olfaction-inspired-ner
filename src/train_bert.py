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
    test_f1 = evaluate(model, test_loader, idx2label, device, args.experiment, verbose=True)
    
    # Save Results
    results = {'test_f1': test_f1, 'best_val_f1': best_f1, 'args': vars(args)}
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
            labels = batch['labels'].to(device) # -100 for ignored

            if experiment_type == 'baseline':
                logits, _ = model(input_ids, mask)
                preds = torch.argmax(logits, dim=2) # (Batch, Seq)
            else:
                # BertOlfactory returns list of paths (decoded tags)
                # We need to map these back or align them
                # Since CRF decode returns LIST of lists of tag INDICES (usually), check BiLSTM_CRF output
                # If existing BiLSTM_CRF returns indices:
                preds_list, _ = model(input_ids, mask) 
                # Pad to tensor for uniform handling or process lists directly
                # Let's assume process lists directly for now
                pass

            # Align Predictions
            # We must only compare tokens where label != -100
            batch_labels = labels.cpu().numpy()
            if experiment_type == 'baseline':
                batch_preds = preds.cpu().numpy()
            else:
                # preds_list is list of list of ints
                batch_preds = preds_list

            for i in range(len(batch_labels)):
                params_labels = []
                params_preds = []
                for j, label_id in enumerate(batch_labels[i]):
                    if label_id != -100:
                        params_labels.append(idx2label[label_id])
                        
                        # Get prediction
                        if experiment_type == 'baseline':
                            pred_id = batch_preds[i][j]
                        else:
                            # CRF output usually matches sequence length logic
                            # But CRF decode might just return the valid path. 
                            # Need to make sure lengths align. 
                            # Usually CRF Viterbi decodes the whole sequence.
                            if j < len(batch_preds[i]):
                                pred_id = batch_preds[i][j]
                            else:
                                pred_id = 0 # Default O
                        
                        params_preds.append(idx2label.get(pred_id, 'O'))
                
                true_labels.append(params_labels)
                pred_labels.append(params_preds)

    return f1_score(true_labels, pred_labels)

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
