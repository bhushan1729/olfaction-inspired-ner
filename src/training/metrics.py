"""
Comprehensive NER metrics utilities.
"""

import numpy as np
from seqeval.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    classification_report,
    accuracy_score
)
from scipy import stats
from typing import List, Dict, Tuple
import json


def compute_ner_metrics(true_labels: List[List[str]], 
                       pred_labels: List[List[str]], 
                       verbose: bool = False) -> Dict:
    """
    Compute comprehensive NER metrics.
    
    Args:
        true_labels: List of true label sequences
        pred_labels: List of predicted label sequences
        verbose: If True, print classification report
    
    Returns:
        Dictionary with all metrics
    """
    # Overall metrics
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Classification report (per-entity breakdown)
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    if verbose:
        print("\n" + "="*80)
        print("NER EVALUATION METRICS")
        print("="*80)
        print(f"Overall F1:        {f1:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall:    {recall:.4f}")
        print(f"Accuracy:          {accuracy:.4f}")
        print("\n" + "-"*80)
        print("Per-Entity Metrics:")
        print("-"*80)
        print(classification_report(true_labels, pred_labels))
    
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'per_entity': report,
        'micro_avg_f1': report.get('micro avg', {}).get('f1-score', f1),
        'macro_avg_f1': report.get('macro avg', {}).get('f1-score', 0.0),
        'weighted_avg_f1': report.get('weighted avg', {}).get('f1-score', 0.0)
    }
    
    return metrics


def get_entity_metrics(metrics: Dict) -> Dict[str, Dict]:
    """
    Extract per-entity metrics from classification report.
    
    Args:
        metrics: Output from compute_ner_metrics
    
    Returns:
        Dictionary mapping entity type to its metrics
    """
    entity_metrics = {}
    
    for key, value in metrics['per_entity'].items():
        # Skip aggregated metrics
        if key in ['micro avg', 'macro avg', 'weighted avg']:
            continue
        
        if isinstance(value, dict):
            entity_metrics[key] = {
                'precision': value.get('precision', 0.0),
                'recall': value.get('recall', 0.0),
                'f1-score': value.get('f1-score', 0.0),
                'support': value.get('support', 0)
            }
    
    return entity_metrics


def compare_predictions(true_labels: List[List[str]], 
                       baseline_preds: List[List[str]], 
                       olfactory_preds: List[List[str]]) -> Dict:
    """
    Compare baseline vs olfactory predictions.
    
    Args:
        true_labels: Ground truth
        baseline_preds: Baseline model predictions
        olfactory_preds: Olfactory model predictions
    
    Returns:
        Comparison statistics
    """
    # Flatten for analysis
    true_flat = [label for seq in true_labels for label in seq]
    base_flat = [label for seq in baseline_preds for label in seq]
    olf_flat = [label for seq in olfactory_preds for label in seq]
    
    # Agreement analysis
    total = len(true_flat)
    both_correct = sum(1 for t, b, o in zip(true_flat, base_flat, olf_flat) 
                      if b == t and o == t)
    only_baseline = sum(1 for t, b, o in zip(true_flat, base_flat, olf_flat) 
                       if b == t and o != t)
    only_olfactory = sum(1 for t, b, o in zip(true_flat, base_flat, olf_flat) 
                        if b != t and o == t)
    both_wrong = sum(1 for t, b, o in zip(true_flat, base_flat, olf_flat) 
                    if b != t and o != t)
    
    return {
        'total_tokens': total,
        'both_correct': both_correct,
        'both_correct_pct': both_correct / total * 100,
        'only_baseline_correct': only_baseline,
        'only_baseline_pct': only_baseline / total * 100,
        'only_olfactory_correct': only_olfactory,
        'only_olfactory_pct': only_olfactory / total * 100,
        'both_wrong': both_wrong,
        'both_wrong_pct': both_wrong / total * 100,
        'olfactory_advantage': only_olfactory - only_baseline,
        'olfactory_advantage_pct': (only_olfactory - only_baseline) / total * 100
    }


def statistical_significance(baseline_scores: List[float], 
                            olfactory_scores: List[float],
                            test: str = 'both') -> Dict:
    """
    Test statistical significance of improvement.
    
    Args:
        baseline_scores: List of baseline F1 scores (e.g., per fold or dataset)
        olfactory_scores: List of olfactory F1 scores
        test: 'paired_t', 'wilcoxon', or 'both'
    
    Returns:
        Test statistics and p-values
    """
    results = {}
    
    if test in ['paired_t', 'both']:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(olfactory_scores, baseline_scores)
        results['paired_t_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Olfactory significantly better' if p_value < 0.05 and t_stat > 0 
                            else 'No significant difference'
        }
    
    if test in ['wilcoxon', 'both']:
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, p_value = stats.wilcoxon(olfactory_scores, baseline_scores)
        results['wilcoxon_test'] = {
            'w_statistic': w_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Olfactory significantly better' if p_value < 0.05 
                            else 'No significant difference'
        }
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(olfactory_scores) - np.mean(baseline_scores)
    pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(olfactory_scores)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    results['effect_size'] = {
        'cohens_d': cohens_d,
        'interpretation': (
            'Large' if abs(cohens_d) >= 0.8 else
            'Medium' if abs(cohens_d) >= 0.5 else
            'Small' if abs(cohens_d) >= 0.2 else
            'Negligible'
        )
    }
    
    return results


def format_metrics_table(baseline_metrics: Dict, 
                         olfactory_metrics: Dict,
                         dataset_name: str) -> str:
    """
    Format metrics as a comparison table.
    
    Args:
        baseline_metrics: Baseline metrics
        olfactory_metrics: Olfactory metrics
        dataset_name: Name of dataset
    
    Returns:
        Formatted table string
    """
    table = f"\n{'='*80}\n"
    table += f"Results: {dataset_name}\n"
    table += f"{'='*80}\n"
    table += f"{'Metric':<20} {'Baseline':<15} {'Olfactory':<15} {'Diff':<15}\n"
    table += f"{'-'*80}\n"
    
    metrics_to_compare = ['f1', 'precision', 'recall', 'accuracy']
    
    for metric in metrics_to_compare:
        base_val = baseline_metrics.get(metric, 0.0)
        olf_val = olfactory_metrics.get(metric, 0.0)
        diff = olf_val - base_val
        diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
        
        table += f"{metric.capitalize():<20} {base_val:<15.4f} {olf_val:<15.4f} {diff_str:<15}\n"
    
    table += f"{'='*80}\n"
    
    return table


def save_metrics(metrics: Dict, filepath: str):
    """Save metrics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Test metrics functions
    true = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC'], ['O', 'O', 'B-ORG']]
    pred = [['O', 'B-PER', 'O', 'O', 'B-LOC'], ['O', 'O', 'B-ORG']]
    
    metrics = compute_ner_metrics(true, pred, verbose=True)
    print("\nEntity metrics:")
    print(json.dumps(get_entity_metrics(metrics), indent=2))
