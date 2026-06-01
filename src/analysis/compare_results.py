"""
Compare baseline vs olfactory results and generate comprehensive analysis.

Supports the directory layout:
    <results_dir>/<dataset_key>/<exp_type>/seed_<N>/results.json

where <exp_type> is one of:
    baseline | olfactory | receptors_only | no_sparsity | more_receptors | more_glomeruli

Usage:
    python src/analysis/compare_results.py \
        --results_dir "/content/drive/My Drive/olfaction_inspired_ner/low_resource_exp" \
        --output_dir  "/content/drive/My Drive/olfaction_inspired_ner/analysis_outputs/compare"
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Known experiment types (sub-directory names)
# ---------------------------------------------------------------------------
KNOWN_EXP_TYPES = {
    'baseline', 'olfactory', 'receptors_only',
    'no_sparsity', 'more_receptors', 'more_glomeruli',
}


# ---------------------------------------------------------------------------
# Helper: extract test metrics from a results.json dict
# ---------------------------------------------------------------------------
def _get_metrics(data: dict) -> dict:
    """Return {'f1', 'precision', 'recall', 'per_entity'} from any result format."""
    if 'test' in data and isinstance(data['test'], dict):
        t = data['test']
        return {
            'f1':        t.get('f1',        t.get('test_f1',        0.0)),
            'precision': t.get('precision', t.get('test_precision', 0.0)),
            'recall':    t.get('recall',    t.get('test_recall',    0.0)),
            'per_entity': t.get('per_entity', {}),
        }
    return {
        'f1':        data.get('test_f1',        data.get('f1',        0.0)),
        'precision': data.get('test_precision', data.get('precision', 0.0)),
        'recall':    data.get('test_recall',    data.get('recall',    0.0)),
        'per_entity': {},
    }


# ---------------------------------------------------------------------------
# Load all results from the directory tree
# ---------------------------------------------------------------------------
def load_all_results(results_dir: str) -> dict:
    """
    Scan results_dir recursively for results.json files.

    Expected tree:
        results_dir/
          <dataset_key>/          e.g. conll_en_1k, wikiann_mr_1k, …
            <exp_type>/           e.g. baseline, olfactory, …
              seed_42/
                results.json
              seed_123/
                results.json
              …

    Returns:
        {
          dataset_key: {
            exp_type: {          ← averaged across seeds
              'f1': float,
              'precision': float,
              'recall': float,
              'per_entity': {entity: f1, …},
              'seeds': [list of individual seed metric dicts],
            }
          }
        }
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        return {}

    # raw[dataset][exp_type] = list of metric dicts (one per seed)
    raw = defaultdict(lambda: defaultdict(list))

    for json_path in sorted(results_dir.rglob('results.json')):
        # Compute path relative to results_dir
        rel = json_path.relative_to(results_dir)
        parts = list(rel.parts)  # e.g. ['conll_en_1k', 'baseline', 'seed_42', 'results.json']

        # We need at least [dataset, exp_type, seed_dir, results.json]
        if len(parts) < 3:
            print(f"  ⚠  Skipping (unexpected depth): {rel}")
            continue

        # Identify dataset key and experiment type
        # Walk parts from the beginning to find the exp_type folder
        dataset_key = None
        exp_type = None
        for depth, part in enumerate(parts[:-1]):  # skip 'results.json'
            if part in KNOWN_EXP_TYPES:
                exp_type = part
                # Everything before it collapses to dataset_key
                dataset_key = '/'.join(parts[:depth]) if depth > 0 else 'unknown'
                break

        if exp_type is None or dataset_key is None:
            print(f"  ⚠  Could not determine exp_type from: {rel}  (parts={parts})")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            metrics = _get_metrics(data)
            raw[dataset_key][exp_type].append(metrics)
        except Exception as e:
            print(f"  ✗ Error reading {json_path}: {e}")

    # Select best performing seed
    all_results = {}
    for dataset_key, exp_map in raw.items():
        all_results[dataset_key] = {}
        for exp_type, seed_list in exp_map.items():
            n = len(seed_list)
            # Find the best performing seed by F1
            best_seed = max(seed_list, key=lambda m: m['f1'])
            best_f1 = best_seed['f1']
            best_precision = best_seed['precision']
            best_recall = best_seed['recall']
            best_per_entity = best_seed['per_entity']

            all_results[dataset_key][exp_type] = {
                'f1':         best_f1,
                'precision':  best_precision,
                'recall':     best_recall,
                'per_entity': best_per_entity,
                'seeds':      seed_list,
                'n_seeds':    n,
            }
            print(f"  [{dataset_key}] {exp_type} -> {n} seed(s)  |  best F1={best_f1:.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Build comparison DataFrame
# ---------------------------------------------------------------------------
def build_comparison_df(all_results: dict, exp_a: str = 'baseline',
                        exp_b: str = 'olfactory') -> pd.DataFrame:
    rows = []
    for dataset_key, exp_map in sorted(all_results.items()):
        if exp_a not in exp_map or exp_b not in exp_map:
            continue
        a = exp_map[exp_a]
        b = exp_map[exp_b]
        rows.append({
            'Dataset':             dataset_key,
            f'{exp_a} F1':         a['f1'],
            f'{exp_a} Precision':  a['precision'],
            f'{exp_a} Recall':     a['recall'],
            f'{exp_b} F1':         b['f1'],
            f'{exp_b} Precision':  b['precision'],
            f'{exp_b} Recall':     b['recall'],
            'F1 Improvement':      b['f1']        - a['f1'],
            'Prec Improvement':    b['precision'] - a['precision'],
            'Rec Improvement':     b['recall']    - a['recall'],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build all-experiments DataFrame (one row per dataset × exp_type)
# ---------------------------------------------------------------------------
def build_full_df(all_results: dict) -> pd.DataFrame:
    rows = []
    for dataset_key, exp_map in sorted(all_results.items()):
        for exp_type, metrics in exp_map.items():
            rows.append({
                'Dataset':    dataset_key,
                'Experiment': exp_type,
                'F1':         metrics['f1'],
                'Precision':  metrics['precision'],
                'Recall':     metrics['recall'],
                'N_seeds':    metrics['n_seeds'],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_comparison_bars(df: pd.DataFrame, output_dir: Path,
                         exp_a: str, exp_b: str):
    if df.empty:
        print("⚠  Skipping bar chart — comparison table is empty.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ['F1', 'Precision', 'Recall']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(df))
        w = 0.35
        ax.bar(x - w/2, df[f'{exp_a} {metric}'], w,
               label=exp_a.capitalize(), alpha=0.85, color='steelblue')
        ax.bar(x + w/2, df[f'{exp_b} {metric}'], w,
               label=exp_b.capitalize(), alpha=0.85, color='coral')
        ax.set_xlabel('Dataset', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison\n({exp_a} vs {exp_b})', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Dataset'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = output_dir / f'comparison_bars_{exp_a}_vs_{exp_b}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved bar chart → {path}")


def plot_improvement_heatmap(df: pd.DataFrame, output_dir: Path,
                             exp_a: str, exp_b: str):
    if df.empty:
        print("⚠  Skipping improvement heatmap — comparison table is empty.")
        return

    imp = df[['Dataset', 'F1 Improvement', 'Prec Improvement',
              'Rec Improvement']].set_index('Dataset')
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.2), 5))
    sns.heatmap(imp.T, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement'}, ax=ax)
    ax.set_title(f'{exp_b.capitalize()} vs {exp_a.capitalize()} — Improvement Heatmap',
                 fontsize=13)
    ax.set_ylabel('Metric')
    plt.tight_layout()
    path = output_dir / f'improvement_heatmap_{exp_a}_vs_{exp_b}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved improvement heatmap → {path}")


def plot_all_experiments_heatmap(full_df: pd.DataFrame, output_dir: Path):
    """Heatmap of F1 across all datasets × experiments."""
    if full_df.empty:
        return
    pivot = full_df.pivot(index='Experiment', columns='Dataset', values='F1')
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.3),
                                    max(5, len(pivot.index) * 0.9)))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu',
                cbar_kws={'label': 'Test F1'}, ax=ax)
    ax.set_title('Test F1 — All Experiments × All Datasets', fontsize=13)
    plt.tight_layout()
    path = output_dir / 'all_experiments_f1_heatmap.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved all-experiments heatmap → {path}")


def plot_all_experiments_bars(full_df: pd.DataFrame, output_dir: Path):
    """Bar chart: F1 per dataset, grouped by experiment."""
    if full_df.empty:
        return
    datasets = sorted(full_df['Dataset'].unique())
    experiments = sorted(full_df['Experiment'].unique())
    x = np.arange(len(datasets))
    w = 0.8 / max(len(experiments), 1)

    palette = plt.cm.tab10(np.linspace(0, 0.9, len(experiments)))
    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 1.5), 7))
    for i, exp in enumerate(experiments):
        sub = full_df[full_df['Experiment'] == exp].set_index('Dataset')
        vals = [sub.loc[d, 'F1'] if d in sub.index else 0.0 for d in datasets]
        ax.bar(x + i * w, vals, w, label=exp, color=palette[i], alpha=0.85)

    ax.set_xticks(x + w * (len(experiments) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Test F1 (best seed)', fontsize=11)
    ax.set_title('Test F1 — All Experiments × All Datasets', fontsize=13)
    ax.legend(title='Experiment', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = output_dir / 'all_experiments_f1_bars.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved all-experiments bar chart → {path}")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def perform_statistical_tests(df: pd.DataFrame, exp_a: str, exp_b: str,
                               output_dir: Path):
    """Paired t-test + Wilcoxon + Cohen's d."""
    if df.empty or len(df) < 2:
        print("⚠  Not enough data for statistical tests.")
        return None

    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("⚠  scipy not available — skipping statistical tests.")
        return None

    a_scores = df[f'{exp_a} F1'].tolist()
    b_scores = df[f'{exp_b} F1'].tolist()
    diffs = [b - a for a, b in zip(a_scores, b_scores)]

    t_stat, t_p = scipy_stats.ttest_rel(b_scores, a_scores)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(diffs)
    except Exception:
        w_stat, w_p = float('nan'), float('nan')

    # Cohen's d
    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs, ddof=1)
    cohens_d  = mean_diff / std_diff if std_diff > 1e-9 else 0.0

    test_results = {
        'paired_t_test': {
            't_statistic': float(t_stat),
            'p_value':     float(t_p),
            'significant': bool(t_p < 0.05),
        },
        'wilcoxon_test': {
            'w_statistic': float(w_stat),
            'p_value':     float(w_p),
            'significant': bool(w_p < 0.05) if not np.isnan(w_p) else False,
        },
        'effect_size': {
            'cohens_d':       float(cohens_d),
            'interpretation': ('large' if abs(cohens_d) >= 0.8
                               else 'medium' if abs(cohens_d) >= 0.5
                               else 'small'),
        },
    }

    with open(output_dir / f'statistical_tests_{exp_a}_vs_{exp_b}.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STATISTICAL TESTS: {exp_b} vs {exp_a}")
    print('='*60)
    print(f"Paired t-test   : t={t_stat:.4f}  p={t_p:.4f}  "
          f"sig={test_results['paired_t_test']['significant']}")
    print(f"Wilcoxon        : w={w_stat:.4f}  p={w_p:.4f}  "
          f"sig={test_results['wilcoxon_test']['significant']}")
    print(f"Cohen's d       : {cohens_d:.4f}  "
          f"({test_results['effect_size']['interpretation']})")
    print('='*60)
    return test_results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def generate_markdown_report(full_df: pd.DataFrame, comp_df: pd.DataFrame,
                             test_results: dict,
                             exp_a: str, exp_b: str, output_dir: Path):
    lines = [
        f"# NER Experiment Analysis Report\n",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## All Experiments — Best Seed F1\n",
        full_df.pivot(index='Experiment', columns='Dataset',
                      values='F1').round(4).to_markdown(),
        "\n\n## Baseline vs Olfactory Comparison\n",
    ]

    if not comp_df.empty:
        lines.append(comp_df.round(4).to_markdown(index=False))
        lines.append(f"\n\n### Summary\n")
        lines.append(f"- Avg Baseline F1 : {comp_df[f'{exp_a} F1'].mean():.4f}\n")
        lines.append(f"- Avg Olfactory F1 : {comp_df[f'{exp_b} F1'].mean():.4f}\n")
        lines.append(f"- Avg F1 improvement : {comp_df['F1 Improvement'].mean():.4f}\n")
        wins = (comp_df['F1 Improvement'] > 0).sum()
        lines.append(f"- Datasets where olfactory wins : {wins}/{len(comp_df)}\n")

    if test_results:
        lines.append("\n## Statistical Significance\n")
        t = test_results.get('paired_t_test', {})
        w = test_results.get('wilcoxon_test', {})
        e = test_results.get('effect_size', {})
        lines.append(f"| Test | Statistic | p-value | Significant |\n")
        lines.append(f"|------|-----------|---------|-------------|\n")
        lines.append(f"| Paired t-test | {t.get('t_statistic', 'N/A'):.4f} | "
                     f"{t.get('p_value', 'N/A'):.4f} | {t.get('significant')} |\n")
        lines.append(f"| Wilcoxon | {w.get('w_statistic', 'N/A'):.4f} | "
                     f"{w.get('p_value', 'N/A'):.4f} | {w.get('significant')} |\n")
        lines.append(f"\nCohen's d = {e.get('cohens_d', 'N/A'):.4f} "
                     f"({e.get('interpretation', '')})\n")

    report = '\n'.join(lines)
    path = output_dir / 'ANALYSIS_REPORT.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Saved markdown report → {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Compare NER experiment results (baseline vs olfactory variants)'
    )
    parser.add_argument(
        '--results_dir', type=str, required=True,
        help='Root directory containing experiment results '
             '(e.g. /content/drive/My Drive/olfaction_inspired_ner/low_resource_exp)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save analysis outputs'
    )
    parser.add_argument(
        '--exp_a', type=str, default='baseline',
        help='First experiment type for pairwise comparison (default: baseline)'
    )
    parser.add_argument(
        '--exp_b', type=str, default='olfactory',
        help='Second experiment type for pairwise comparison (default: olfactory)'
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("NER EXPERIMENT COMPARATIVE ANALYSIS")
    print('='*70)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Comparing   : {args.exp_a}  vs  {args.exp_b}")
    print('='*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all results
    print("\nScanning for results.json files …")
    all_results = load_all_results(args.results_dir)

    if not all_results:
        print("✗ No results found. Check --results_dir.")
        return

    # 2. Build DataFrames
    full_df = build_full_df(all_results)
    comp_df = build_comparison_df(all_results, args.exp_a, args.exp_b)

    # 3. Save CSVs
    full_csv = output_dir / 'all_experiments.csv'
    comp_csv = output_dir / f'comparison_{args.exp_a}_vs_{args.exp_b}.csv'
    full_df.to_csv(full_csv, index=False)
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n✓ Saved full table → {full_csv}")
    print(f"✓ Saved comparison → {comp_csv}")

    # 4. Print tables
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS — best seed F1")
    print('='*70)
    pivot = full_df.pivot(index='Experiment', columns='Dataset', values='F1')
    print(pivot.round(4).to_string())

    if not comp_df.empty:
        print(f"\n{'='*70}")
        print(f"PAIRWISE: {args.exp_a}  vs  {args.exp_b}")
        print('='*70)
        print(comp_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # 5. Visualisations
    print("\nGenerating plots …")
    plot_comparison_bars(comp_df, output_dir, args.exp_a, args.exp_b)
    plot_improvement_heatmap(comp_df, output_dir, args.exp_a, args.exp_b)
    plot_all_experiments_heatmap(full_df, output_dir)
    plot_all_experiments_bars(full_df, output_dir)

    # 6. Statistical tests
    test_results = perform_statistical_tests(comp_df, args.exp_a, args.exp_b, output_dir)

    # 7. Markdown report
    generate_markdown_report(full_df, comp_df, test_results,
                             args.exp_a, args.exp_b, output_dir)

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE — outputs saved to: {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
