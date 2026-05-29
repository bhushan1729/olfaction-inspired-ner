"""
Receptor & glomeruli activation visualizations for olfactory NER models.

For each (dataset, experiment-type) only the **best seed** (highest test F1)
is used to generate:
  1. Receptor activation heatmap (entities × receptor units)
  2. Glomeruli activation heatmap (entities × glomerulus units)
  3. RSI (Receptor Selectivity Index) histogram — receptors & glomeruli
  4. t-SNE of glomerular representations coloured by entity type
  5. Top-activating tokens per receptor (saved as JSON)

Expected directory layout:
    <results_dir>/<dataset_key>/<exp_type>/seed_<N>/
        results.json
        best_model.pt

Usage:
    python src/analysis/visualize.py \
        --results_dir "/content/drive/My Drive/olfaction_inspired_ner/low_resource_exp" \
        --output_dir  "/content/drive/My Drive/olfaction_inspired_ner/analysis_outputs/visualize" \
        --data_dir    "/content/drive/My Drive/olfaction_inspired_ner/data"
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
KNOWN_EXP_TYPES = {
    'baseline', 'olfactory', 'receptors_only',
    'no_sparsity', 'more_receptors', 'more_glomeruli',
}

OLFACTORY_EXP_TYPES = KNOWN_EXP_TYPES - {'baseline'}


# ---------------------------------------------------------------------------
# Scanning — identical logic as generate_heatmaps.py
# ---------------------------------------------------------------------------
def _test_f1(json_path: Path) -> float:
    try:
        with open(json_path, 'r') as f:
            d = json.load(f)
        t = d.get('test', {})
        return float(t.get('f1', t.get('test_f1', 0.0)))
    except Exception:
        return 0.0


def find_best_seeds(results_dir: Path) -> list:
    """Return one entry per (dataset, exp_type) with the best-F1 seed."""
    raw = defaultdict(lambda: defaultdict(list))

    for json_path in sorted(results_dir.rglob('results.json')):
        rel   = json_path.relative_to(results_dir)
        parts = list(rel.parts)

        dataset_key = None
        exp_type    = None
        for depth, part in enumerate(parts[:-1]):
            if part in KNOWN_EXP_TYPES:
                exp_type    = part
                dataset_key = '/'.join(parts[:depth]) if depth > 0 else 'unknown'
                break

        if exp_type is None or dataset_key is None:
            continue

        f1 = _test_f1(json_path)
        raw[dataset_key][exp_type].append((f1, json_path.parent, json_path))

    best = []
    for dataset_key, exp_map in sorted(raw.items()):
        for exp_type, candidates in sorted(exp_map.items()):
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_f1, best_seed_dir, best_json = candidates[0]
            config = {}
            try:
                with open(best_json) as f:
                    config = json.load(f).get('config', {})
            except Exception:
                pass

            model_path = best_seed_dir / 'best_model.pt'
            print(f"  [{dataset_key}] {exp_type:20s}  "
                  f"seed={best_seed_dir.name}  F1={best_f1:.4f}  "
                  f"model={'✓' if model_path.exists() else '✗'}")
            best.append({
                'dataset':    dataset_key,
                'exp_type':   exp_type,
                'seed_dir':   best_seed_dir,
                'model_path': model_path,
                'config':     config,
                'test_f1':    best_f1,
            })
    return best


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------
def collect_activations(model, data_loader, device, idx2label: dict,
                        idx2word: dict) -> dict:
    """
    Run model through test_loader and return raw per-token activations.

    Returns dict with keys:
        'receptor':   {entity_type: list of 1-D numpy arrays}
        'glomeruli':  {entity_type: list of 1-D numpy arrays}
        'token_receptor': {receptor_idx: [(token_str, activation_val), …]}
    """
    model.eval()
    receptor_by_entity  = defaultdict(list)
    glomeruli_by_entity = defaultdict(list)
    token_receptor      = defaultdict(list)   # receptor_idx → [(token, act)]

    with torch.no_grad():
        for batch in data_loader:
            sentences, tags, lengths = batch
            sentences = sentences.to(device)
            tags      = tags.to(device)
            lengths   = lengths.to(device)

            try:
                receptors, glomeruli, _ = model.get_receptor_activations(sentences)
            except Exception:
                return {
                    'receptor':       {},
                    'glomeruli':      {},
                    'token_receptor': {},
                }

            for i, length in enumerate(lengths):
                L = int(length.item())
                for t in range(L):
                    label = idx2label.get(int(tags[i, t].item()), 'O')
                    if label != 'O':
                        entity_type = label.split('-', 1)[-1]
                        if receptors is not None:
                            arr = receptors[i, t].cpu().numpy()
                            receptor_by_entity[entity_type].append(arr)
                            # track top token→receptor activations
                            token_str = idx2word.get(
                                int(sentences[i, t].item()), '<unk>')
                            for r_idx, act in enumerate(arr):
                                if act > 0.1:
                                    token_receptor[r_idx].append(
                                        (token_str, float(act)))
                        if glomeruli is not None:
                            glomeruli_by_entity[entity_type].append(
                                glomeruli[i, t].cpu().numpy())

    return {
        'receptor':       dict(receptor_by_entity),
        'glomeruli':      dict(glomeruli_by_entity),
        'token_receptor': dict(token_receptor),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _mean_matrix(by_entity: dict, entity_types: list) -> np.ndarray:
    """[n_entities, n_units] mean activation matrix."""
    return np.array([
        np.mean(by_entity[e], axis=0)
        for e in entity_types if e in by_entity
    ])


def plot_heatmap(matrix: np.ndarray, entity_types: list, layer_name: str,
                 title: str, save_path: Path):
    if matrix.size == 0:
        return
    n_units = matrix.shape[1]
    step    = max(1, n_units // 32)
    xlabels = [str(i) if i % step == 0 else '' for i in range(n_units)]

    fig, ax = plt.subplots(figsize=(max(14, n_units // 4),
                                    max(4, len(entity_types) + 1)))
    sns.heatmap(matrix,
                yticklabels=entity_types,
                xticklabels=xlabels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Mean Activation'},
                ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(f'{layer_name} Unit Index', fontsize=11)
    ax.set_ylabel('Entity Type', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Heatmap → {save_path.name}")


def compute_rsi(mean_matrix: np.ndarray) -> list:
    """Compute Receptor Selectivity Index per unit."""
    rsi = []
    for u in range(mean_matrix.shape[1]):
        mus    = mean_matrix[:, u]
        mx, mn = mus.max(), mus.min()
        rsi.append(float((mx - mn) / mx) if mx > 1e-6 else 0.0)
    return rsi


def plot_rsi_histogram(rsi_scores: list, layer_name: str,
                       title: str, save_path: Path, color: str = 'purple'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rsi_scores, bins=20, color=color, alpha=0.75, edgecolor='black')
    ax.set_xlabel(f'{layer_name} Selectivity Index (RSI)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ RSI histogram → {save_path.name}")


def plot_tsne(glomeruli_by_entity: dict, entity_types: list,
              title: str, save_path: Path, max_per_entity: int = 200):
    from sklearn.manifold import TSNE

    all_vecs, all_labels = [], []
    for entity in entity_types:
        acts = glomeruli_by_entity.get(entity, [])
        n    = min(max_per_entity, len(acts))
        if n == 0:
            continue
        all_vecs.extend(acts[:n])
        all_labels.extend([entity] * n)

    if not all_vecs:
        print("  ⚠  No glomeruli activations for t-SNE — skipping.")
        return

    X = np.array(all_vecs)
    perp = min(30, max(5, len(X) // 5))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    X2   = tsne.fit_transform(X)

    palette = plt.cm.tab10(np.linspace(0, 0.9, len(entity_types)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, entity in enumerate(entity_types):
        idx = [j for j, lb in enumerate(all_labels) if lb == entity]
        if not idx:
            continue
        ax.scatter(X2[idx, 0], X2[idx, 1],
                   c=[palette[i]], label=entity,
                   alpha=0.65, s=40, edgecolors='none')

    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(title='Entity', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ t-SNE → {save_path.name}")


def save_top_tokens(token_receptor: dict, save_path: Path, top_k: int = 10):
    """Save top-k tokens per receptor index as JSON."""
    out = {}
    for r_idx, pairs in token_receptor.items():
        top = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
        out[int(r_idx)] = [{'token': t, 'activation': a} for t, a in top]

    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Top tokens → {save_path.name}")


# ---------------------------------------------------------------------------
# Process one (dataset, exp_type) best seed
# ---------------------------------------------------------------------------
def process_entry(entry: dict, vocab_info: dict,
                  test_loader, device, output_dir: Path):
    dataset  = entry['dataset']
    exp_type = entry['exp_type']
    config   = entry['config']
    model_type = config.get('model_type', '')

    if model_type not in OLFACTORY_EXP_TYPES and exp_type not in OLFACTORY_EXP_TYPES:
        print(f"  ⚠  [{dataset}] {exp_type}: not olfactory — skipping.")
        return

    model_path = entry['model_path']
    if not model_path.exists():
        print(f"  ⚠  [{dataset}] {exp_type}: best_model.pt missing — skipping.")
        return

    print(f"\n[{dataset}] {exp_type}  (F1={entry['test_f1']:.4f}  seed={entry['seed_dir'].name})")

    # Load checkpoint
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ✗ Cannot load checkpoint: {e}")
        return

    state_dict = ckpt.get('model_state_dict', ckpt)

    vocab_size = (state_dict['embedding.weight'].shape[0]
                  if 'embedding.weight' in state_dict
                  else len(vocab_info.get('word2idx', {})))
    if 'hidden2tag.weight' in state_dict:
        num_tags = state_dict['hidden2tag.weight'].shape[0]
    elif 'output_layer.weight' in state_dict:
        num_tags = state_dict['output_layer.weight'].shape[0]
    else:
        num_tags = len(vocab_info.get('label2idx', {}))

    # Label mapping
    if 'label2idx' in ckpt:
        idx2label = {v: k for k, v in ckpt['label2idx'].items()}
    elif 'label2idx' in vocab_info:
        idx2label = {v: k for k, v in vocab_info['label2idx'].items()}
    else:
        idx2label = {i: str(i) for i in range(num_tags)}

    idx2word = vocab_info.get('idx2word', {})

    # Build & load model
    try:
        from src.model.olfactory_ner import create_olfactory_ner
        model = create_olfactory_ner(vocab_size, num_tags, config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  ✗ Cannot instantiate model: {e}")
        return

    # Collect activations
    print("  Collecting activations …")
    acts = collect_activations(model, test_loader, device, idx2label, idx2word)

    # Entity types (excluding aggregate rows)
    skip_entities = {'micro avg', 'macro avg', 'weighted avg', 'O'}
    entity_types = sorted(
        (e for e in set(list(acts['receptor'].keys()) +
                        list(acts['glomeruli'].keys()))
         if e not in skip_entities)
    )
    if not entity_types:
        print("  ⚠  No entity activations collected — skipping.")
        return

    # Output file prefix
    safe_ds = dataset.replace('/', '_')
    prefix  = output_dir / f"{safe_ds}__{exp_type}"
    f1_tag  = f"F1={entry['test_f1']:.4f}"

    # ── 1. Receptor heatmap ─────────────────────────────────────────────
    if acts['receptor']:
        valid = [e for e in entity_types if e in acts['receptor']]
        r_matrix = _mean_matrix(acts['receptor'], valid)
        plot_heatmap(
            r_matrix, valid, 'Receptor',
            title=f"Receptor Activations — {dataset} / {exp_type}  ({f1_tag})",
            save_path=Path(str(prefix) + '__receptor_heatmap.png'),
        )

        # RSI
        rsi_r = compute_rsi(r_matrix)
        print(f"  Receptor avg RSI: {np.mean(rsi_r):.4f}")
        plot_rsi_histogram(
            rsi_r, 'Receptor',
            title=f"Receptor RSI — {dataset} / {exp_type}  ({f1_tag})",
            save_path=Path(str(prefix) + '__receptor_rsi.png'),
            color='purple',
        )

    # ── 2. Glomeruli heatmap ────────────────────────────────────────────
    if acts['glomeruli']:
        valid = [e for e in entity_types if e in acts['glomeruli']]
        g_matrix = _mean_matrix(acts['glomeruli'], valid)
        plot_heatmap(
            g_matrix, valid, 'Glomerulus',
            title=f"Glomeruli Activations — {dataset} / {exp_type}  ({f1_tag})",
            save_path=Path(str(prefix) + '__glomeruli_heatmap.png'),
        )

        # RSI
        rsi_g = compute_rsi(g_matrix)
        print(f"  Glomeruli avg RSI: {np.mean(rsi_g):.4f}")
        plot_rsi_histogram(
            rsi_g, 'Glomerulus',
            title=f"Glomeruli RSI — {dataset} / {exp_type}  ({f1_tag})",
            save_path=Path(str(prefix) + '__glomeruli_rsi.png'),
            color='orange',
        )

        # t-SNE
        plot_tsne(
            acts['glomeruli'], valid,
            title=f"t-SNE of Glomeruli — {dataset} / {exp_type}  ({f1_tag})",
            save_path=Path(str(prefix) + '__tsne.png'),
        )

    # ── 3. Top tokens per receptor ──────────────────────────────────────
    if acts['token_receptor']:
        save_top_tokens(
            acts['token_receptor'],
            save_path=Path(str(prefix) + '__top_tokens.json'),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Visualize receptor/glomeruli activations (best-seed per experiment)'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Root dir of experiment results')
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='Where to save output files')
    parser.add_argument('--data_dir',    type=str, required=True,
                        help='Directory with raw dataset files')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--no_cuda',     action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available()
                          else 'cuda')

    print(f"\n{'='*70}")
    print("RECEPTOR / GLOMERULI VISUALIZATIONS — BEST SEED PER EXPERIMENT")
    print('='*70)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Data dir    : {args.data_dir}")
    print(f"Device      : {device}")
    print('='*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading dataset …")
    try:
        from src.data.dataset import prepare_data
        _, _, test_loader, vocab_info = prepare_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            min_freq=2,
        )
        print(f"✓ Test loader ready ({len(test_loader.dataset)} samples)")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return

    # Find best seeds
    print("\nFinding best seed per (dataset, experiment) …")
    results_dir = Path(args.results_dir)
    all_entries = find_best_seeds(results_dir)

    olf_entries = [e for e in all_entries if e['exp_type'] in OLFACTORY_EXP_TYPES]
    print(f"\nProcessing {len(olf_entries)} olfactory-type experiment(s) …")

    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print('='*70)

    for entry in olf_entries:
        process_entry(entry, vocab_info, test_loader, device, output_dir)

    print(f"\n{'='*70}")
    print(f"DONE — outputs saved to: {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
