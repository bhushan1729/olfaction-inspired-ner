"""
Comprehensive analysis & visualizations for all NER experiments.

Scans the results directory for the layout:
    <results_dir>/<dataset_key>/<exp_type>/seed_<N>/results.json

For each (dataset, experiment) combination it computes the mean F1/P/R across
all seeds, prints a formatted table, and produces:
  1. Per-entity F1 bar chart (per dataset)
  2. Precision-Recall bubble chart (per dataset)
  3. Cross-dataset F1 heatmap (all experiments)
  4. Cross-dataset F1 grouped bar chart (all experiments)

Usage:
    python src/analysis/final_analysis.py \
        --results_dir "/content/drive/My Drive/olfaction_inspired_ner/low_resource_exp" \
        --output_dir  "/content/drive/My Drive/olfaction_inspired_ner/analysis_outputs/final"
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Known experiment / dataset names (for smart path parsing)
# ---------------------------------------------------------------------------
KNOWN_EXP_TYPES = {
    'baseline', 'olfactory', 'receptors_only',
    'no_sparsity', 'more_receptors', 'more_glomeruli',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_metrics(data: dict) -> dict:
    """Extract F1/Precision/Recall and per-entity scores from a results.json dict."""
    if 'test' in data and isinstance(data['test'], dict):
        t = data['test']
        return {
            'f1':         t.get('f1',        0.0),
            'precision':  t.get('precision', 0.0),
            'recall':     t.get('recall',    0.0),
            'per_entity': t.get('per_entity', {}),
        }
    return {
        'f1':         data.get('test_f1', data.get('f1', 0.0)),
        'precision':  data.get('test_precision', data.get('precision', 0.0)),
        'recall':     data.get('test_recall',    data.get('recall',    0.0)),
        'per_entity': {},
    }


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def load_all_results(results_dir: str) -> dict:
    """
    Walk results_dir and aggregate metrics across seeds.

    Returns:
        {
          dataset_key: {
            exp_type: {
              'f1', 'precision', 'recall',
              'per_entity': {entity: avg_f1},
              'n_seeds': int,
            }
          }
        }
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"✗ Directory not found: {results_dir}")
        return {}

    # raw[dataset][exp_type] = list of metric dicts
    raw = defaultdict(lambda: defaultdict(list))

    for json_path in sorted(results_dir.rglob('results.json')):
        rel   = json_path.relative_to(results_dir)
        parts = list(rel.parts)          # e.g. ['conll_en_1k', 'baseline', 'seed_42', 'results.json']

        dataset_key = None
        exp_type    = None
        for depth, part in enumerate(parts[:-1]):
            if part in KNOWN_EXP_TYPES:
                exp_type    = part
                dataset_key = '/'.join(parts[:depth]) if depth > 0 else 'unknown'
                break

        if exp_type is None or dataset_key is None:
            print(f"  ⚠  Cannot parse path: {rel}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            raw[dataset_key][exp_type].append(_get_metrics(data))
        except Exception as e:
            print(f"  ✗ Error reading {json_path}: {e}")

    # Average across seeds
    aggregated = {}
    for dataset_key, exp_map in raw.items():
        aggregated[dataset_key] = {}
        for exp_type, seed_list in exp_map.items():
            n  = len(seed_list)
            avg = {
                'f1':        sum(m['f1']        for m in seed_list) / n,
                'precision': sum(m['precision'] for m in seed_list) / n,
                'recall':    sum(m['recall']    for m in seed_list) / n,
                'n_seeds':   n,
            }
            # Average per_entity
            per_e = defaultdict(list)
            for m in seed_list:
                for entity, score in m['per_entity'].items():
                    if isinstance(score, (int, float)):
                        per_e[entity].append(score)
            avg['per_entity'] = {e: sum(v) / len(v) for e, v in per_e.items()}
            aggregated[dataset_key][exp_type] = avg
            print(f"  [{dataset_key}] {exp_type:20s} {n} seed(s)  F1={avg['f1']:.4f}")

    return aggregated


# ---------------------------------------------------------------------------
# Build DataFrames
# ---------------------------------------------------------------------------
def build_full_df(all_results: dict) -> pd.DataFrame:
    rows = []
    for dataset_key, exp_map in sorted(all_results.items()):
        for exp_type, m in exp_map.items():
            row = {
                'Dataset':    dataset_key,
                'Experiment': exp_type,
                'F1':         m['f1'],
                'Precision':  m['precision'],
                'Recall':     m['recall'],
                'N_seeds':    m['n_seeds'],
            }
            for entity, score in m['per_entity'].items():
                row[f'{entity}_F1'] = score
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_per_entity_bars(dataset_key: str, df_dataset: pd.DataFrame,
                         dataset_dir: Path):
    """Bar chart: per-entity F1 grouped by experiment for one dataset."""
    entity_cols = sorted([c for c in df_dataset.columns
                          if c.endswith('_F1') and c not in ('F1',)])
    if not entity_cols:
        return

    df_long = df_dataset.melt(
        id_vars='Experiment', value_vars=entity_cols,
        var_name='Entity Type', value_name='F1 Score'
    )
    df_long['Entity Type'] = df_long['Entity Type'].str.replace('_F1', '', regex=False)

    # Filter out aggregated rows (micro avg, macro avg, weighted avg)
    skip = {'micro avg', 'macro avg', 'weighted avg'}
    df_long = df_long[~df_long['Entity Type'].isin(skip)]

    n_exp = df_dataset['Experiment'].nunique()
    fig, ax = plt.subplots(figsize=(max(10, n_exp * 3), 6))
    palette = sns.color_palette('tab10', n_colors=n_exp)
    sns.barplot(data=df_long, x='Entity Type', y='F1 Score',
                hue='Experiment', palette=palette, edgecolor='black', ax=ax)
    ax.set_title(f'Per-Entity F1 Score — {dataset_key}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Experiment')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = dataset_dir / 'entity_f1.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ entity_f1.png")


def plot_pr_bubble(dataset_key: str, df_dataset: pd.DataFrame, dataset_dir: Path):
    """Precision–Recall bubble chart for one dataset."""
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette('Set2', n_colors=len(df_dataset))

    for i, (_, row) in enumerate(df_dataset.iterrows()):
        ax.scatter(row['Recall'], row['Precision'],
                   s=row['F1'] * 2000,
                   color=palette[i], alpha=0.75,
                   edgecolors='black', linewidths=0.8,
                   label=row['Experiment'])
        ax.text(row['Recall'] + 0.002, row['Precision'] + 0.002,
                row['Experiment'], fontsize=8)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision vs Recall (bubble ∝ F1) — {dataset_key}',
                 fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Experiment')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = dataset_dir / 'pr_bubble.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ pr_bubble.png")


def plot_cross_dataset_heatmap(full_df: pd.DataFrame, output_dir: Path):
    """Heatmap: F1 across all experiments × datasets."""
    pivot = full_df.pivot_table(index='Experiment', columns='Dataset',
                                values='F1', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.3),
                                    max(4, len(pivot.index) * 0.9)))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu',
                cbar_kws={'label': 'Test F1'}, ax=ax,
                linewidths=0.4, linecolor='white')
    ax.set_title('Test F1 — All Experiments × All Datasets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Experiment', fontsize=11)
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = output_dir / 'cross_dataset_f1_heatmap.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved cross-dataset heatmap → {path}")


def plot_cross_dataset_bars(full_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: F1 per dataset × experiment."""
    datasets    = sorted(full_df['Dataset'].unique())
    experiments = sorted(full_df['Experiment'].unique())
    x = np.arange(len(datasets))
    w = 0.8 / max(len(experiments), 1)
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(experiments)))

    fig, ax = plt.subplots(figsize=(max(14, len(datasets) * 1.6), 7))
    for i, exp in enumerate(experiments):
        sub  = full_df[full_df['Experiment'] == exp].set_index('Dataset')
        vals = [sub.loc[d, 'F1'] if d in sub.index else 0.0 for d in datasets]
        ax.bar(x + i * w, vals, w, label=exp, color=palette[i], alpha=0.88)

    ax.set_xticks(x + w * (len(experiments) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Test F1 (mean across seeds)', fontsize=11)
    ax.set_title('Test F1 — All Experiments & Datasets', fontsize=14, fontweight='bold')
    ax.legend(title='Experiment', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = output_dir / 'cross_dataset_f1_bars.png'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cross-dataset bar chart → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Final comprehensive analysis of all NER experiments'
    )
    parser.add_argument(
        '--results_dir', type=str, required=True,
        help='Root directory containing experiment results '
             '(e.g. /content/drive/My Drive/olfaction_inspired_ner/low_resource_exp)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save all analysis outputs'
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("FINAL COMPREHENSIVE NER ANALYSIS")
    print('='*70)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print('='*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scan for results
    print("\nScanning results …")
    all_results = load_all_results(args.results_dir)
    if not all_results:
        print("✗ No results found. Check --results_dir.")
        return

    # 2. Build DataFrame
    full_df = build_full_df(all_results)

    # 3. Save CSV
    csv_path = output_dir / 'all_experiments_summary.csv'
    full_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved summary CSV → {csv_path}")

    # 4. Print tables per dataset
    print(f"\n{'='*70}")
    for dataset_key in sorted(full_df['Dataset'].unique()):
        df_d = full_df[full_df['Dataset'] == dataset_key].copy()
        df_d = df_d.sort_values('F1', ascending=False)
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_key}")
        print('='*70)
        base_cols = ['Experiment', 'F1', 'Precision', 'Recall']
        entity_cols = sorted([c for c in df_d.columns
                               if c.endswith('_F1')
                               and c not in {'F1'}
                               and 'avg' not in c.lower()])
        show_cols = [c for c in base_cols + entity_cols if c in df_d.columns]
        print(df_d[show_cols].round(4).to_string(index=False))

    # 5. Per-dataset plots — saved in output_dir/<dataset>/
    print(f"\n{'='*70}")
    print("Generating per-dataset plots …")
    for dataset_key in sorted(full_df['Dataset'].unique()):
        df_d = full_df[full_df['Dataset'] == dataset_key].copy()
        safe_ds   = dataset_key.replace('/', '_')
        dataset_dir = output_dir / safe_ds
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{dataset_key}] → {dataset_dir}")

        # Per-dataset CSV
        df_d.to_csv(dataset_dir / 'results_table.csv', index=False)

        plot_per_entity_bars(dataset_key, df_d, dataset_dir)
        plot_pr_bubble(dataset_key, df_d, dataset_dir)

    # 6. Cross-dataset plots — saved at output_dir root
    print("\nGenerating cross-dataset plots …")
    plot_cross_dataset_heatmap(full_df, output_dir)
    plot_cross_dataset_bars(full_df, output_dir)

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE — outputs saved to: {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
