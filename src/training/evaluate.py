"""
Evaluation utilities for NER models.
Uses seqeval for proper entity-level metrics.
"""

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm


def evaluate_model(model, data_loader, idx2label, device):
    """
    Evaluate NER model on a dataset.
    
    Args:
        model: NER model
        data_loader: DataLoader for evaluation
        idx2label: Dict mapping tag indices to labels
        device: torch device
    
    Returns:
        metrics: Dict with precision, recall, F1, and per-entity scores
    """
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for sentences, tags, lengths in tqdm(data_loader, desc="Evaluating"):
            sentences = sentences.to(device)
            lengths = lengths.to(device)
            
            # Get predictions
            predictions = model(sentences, lengths=lengths)
            
            # Convert to labels
            for i in range(len(sentences)):
                length = lengths[i].item()
                
                # True labels
                true_labels = [idx2label[tags[i, j].item()] for j in range(length)]
                
                # Predicted labels
                pred_labels = [idx2label[predictions[i, j].item()] for j in range(length)]
                
                all_true_labels.append(true_labels)
                all_predictions.append(pred_labels)
    
    # Compute metrics using seqeval
    f1 = f1_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions)
    recall = recall_score(all_true_labels, all_predictions)
    
    # Get detailed report
    report = classification_report(all_true_labels, all_predictions, output_dict=True)
    
    # Extract per-entity F1 scores
    per_entity_f1 = {}
    for key, value in report.items():
        if isinstance(value, dict) and 'f1-score' in value:
            per_entity_f1[key] = value['f1-score']
    
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'per_entity': per_entity_f1
    }
    
    return metrics


def get_predictions(model, data_loader, idx2word, idx2label, device):
    """
    Get detailed predictions for analysis.
    
    Returns:
        results: List of dicts with tokens, true labels, and predictions
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for sentences, tags, lengths in data_loader:
            sentences = sentences.to(device)
            lengths = lengths.to(device)
            
            predictions = model(sentences, lengths=lengths)
            
            for i in range(len(sentences)):
                length = lengths[i].item()
                
                tokens = [idx2word[sentences[i, j].item()] for j in range(length)]
                true_labels = [idx2label[tags[i, j].item()] for j in range(length)]
                pred_labels = [idx2label[predictions[i, j].item()] for j in range(length)]
                
                results.append({
                    'tokens': tokens,
                    'true_labels': true_labels,
                    'pred_labels': pred_labels
                })
    
    return results


if __name__ == '__main__':
    # Test with dummy data
    from collections import defaultdict
    
    # Create dummy predictions and labels
    true_labels = [
        ['O', 'B-PER', 'I-PER', 'O', 'B-LOC'],
        ['B-ORG', 'I-ORG', 'O', 'O'],
    ]
    
    predictions = [
        ['O', 'B-PER', 'I-PER', 'O', 'B-LOC'],
        ['B-ORG', 'I-ORG', 'B-PER', 'O'],
    ]
    
    # Compute metrics
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("\nDetailed report:")
    print(classification_report(true_labels, predictions))
