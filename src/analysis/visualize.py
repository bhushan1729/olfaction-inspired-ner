"""
Receptor & glomeruli activation visualizations for olfactory NER models.

Output hierarchy:
    <output_dir>/
      <dataset_key>/
        <exp_type>/
          seed_<N>/
            receptor_heatmap.png
            glomeruli_heatmap.png   (if use_glomeruli)
            receptor_rsi.png
            glomeruli_rsi.png
            tsne.png
            top_tokens.json
          receptor_heatmap_all_seeds.png    ← mean across ALL seeds
          glomeruli_heatmap_all_seeds.png
          receptor_rsi_all_seeds.png
          glomeruli_rsi_all_seeds.png
          tsne_all_seeds.png
          best_seed.txt

Each model is evaluated on its OWN test split (dataset/language from results.json).
Test data is loaded once per experiment group to avoid redundant downloads.

Usage:
    python src/analysis/visualize.py \
        --results_dir "/content/drive/My Drive/olfaction_inspired_ner/low_resource_exp" \
        --output_dir  "/content/drive/My Drive/olfaction_inspired_ner/analysis_outputs/visualize" \
        --cache_dir   "/content/drive/My Drive/olfaction_inspired_ner/data"
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
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOWN_EXP_TYPES = {
    'baseline', 'olfactory', 'receptors_only',
    'no_sparsity', 'more_receptors', 'more_glomeruli',
}
OLFACTORY_EXP_TYPES = KNOWN_EXP_TYPES - {'baseline'}
SKIP_ENTITIES = {'micro avg', 'macro avg', 'weighted avg', 'O'}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def _test_f1(json_path: Path) -> float:
    try:
        with open(json_path, 'r') as f:
            d = json.load(f)
        return float(d.get('test', {}).get('f1', 0.0))
    except Exception:
        return 0.0


def find_all_seeds_grouped(results_dir: Path) -> dict:
    """
    Returns {(dataset_key, exp_type): [seed_entries sorted best→worst]}
    where each entry is {seed_dir, model_path, config, test_f1}.
    """
    raw = defaultdict(list)
    for json_path in sorted(results_dir.rglob('results.json')):
        rel   = json_path.relative_to(results_dir)
        parts = list(rel.parts)
        dataset_key = exp_type = None
        for depth, part in enumerate(parts[:-1]):
            if part in KNOWN_EXP_TYPES:
                exp_type    = part
                dataset_key = '/'.join(parts[:depth]) if depth > 0 else 'unknown'
                break
        if exp_type is None:
            continue
        config = {}
        try:
            with open(json_path) as f:
                config = json.load(f).get('config', {})
        except Exception:
            pass
        raw[(dataset_key, exp_type)].append({
            'seed_dir':   json_path.parent,
            'model_path': json_path.parent / 'best_model.pt',
            'config':     config,
            'test_f1':    _test_f1(json_path),
        })

    grouped = {}
    for key, entries in raw.items():
        entries.sort(key=lambda e: e['test_f1'], reverse=True)
        grouped[key] = entries
    return grouped


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_test_loader(config: dict, cache_dir: str, batch_size: int = 32):
    from src.data.unified_loader import get_dataset
    dataset_name = config.get('dataset', 'conll2003')
    language     = config.get('language', None)
    print(f"    Loading data: dataset={dataset_name}  language={language}")
    try:
        _, _, test_loader, vocab_info = get_dataset(
            dataset_name=dataset_name,
            language=language,
            cache_dir=cache_dir,
            batch_size=batch_size,
            min_freq=config.get('min_freq', 2),
            max_train_samples=config.get('max_train_samples', None),
        )
        print(f"    ✓ {len(test_loader.dataset)} test samples")
        return test_loader, vocab_info
    except Exception as e:
        print(f"    ✗ Data load failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(entry: dict, num_tags: int, device):
    from src.model.olfactory_ner import create_olfactory_ner
    if not entry['model_path'].exists():
        print(f"      ✗ best_model.pt not found")
        return None
    try:
        ckpt       = torch.load(entry['model_path'], map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        vocab_size = (state_dict['embedding.weight'].shape[0]
                      if 'embedding.weight' in state_dict else 0)
        model = create_olfactory_ner(vocab_size, num_tags, entry['config'])
        model.load_state_dict(state_dict, strict=False)
        return model.to(device).eval()
    except Exception as e:
        print(f"      ✗ Model load error: {e}")
        return None


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------
def collect_activations(model, data_loader, device,
                        idx2label: dict, idx2word: dict) -> dict:
    """
    Returns:
        receptor       : {entity: [array_per_token, …]}
        glomeruli      : {entity: [array_per_token, …]}
        token_receptor : {receptor_idx: [(token, activation), …]}
    """
    receptor_by_entity  = defaultdict(list)
    glomeruli_by_entity = defaultdict(list)
    token_receptor      = defaultdict(list)
    first_exc = None
    n_good    = 0

    model.eval()
    with torch.no_grad():
        for sentences, tags, lengths in data_loader:
            sentences = sentences.to(device)
            tags      = tags.to(device)
            lengths   = lengths.to(device)
            try:
                receptors, glomeruli, _ = model.get_receptor_activations(sentences)
            except Exception as e:
                if first_exc is None:
                    first_exc = e
                continue
            n_good += 1
            for i, length in enumerate(lengths):
                for t in range(int(length.item())):
                    label = idx2label.get(int(tags[i, t].item()), 'O')
                    if label == 'O':
                        continue
                    entity = label.split('-', 1)[-1]
                    if receptors is not None:
                        arr = receptors[i, t].cpu().numpy()
                        receptor_by_entity[entity].append(arr)
                        tok = idx2word.get(int(sentences[i, t].item()), '<unk>')
                        for r_idx, act in enumerate(arr):
                            if act > 0.1:
                                token_receptor[r_idx].append((tok, float(act)))
                    if glomeruli is not None:
                        glomeruli_by_entity[entity].append(
                            glomeruli[i, t].cpu().numpy())

    if first_exc and n_good == 0:
        print(f"      ✗ get_receptor_activations failed: {first_exc}")

    return {
        'receptor':       dict(receptor_by_entity),
        'glomeruli':      dict(glomeruli_by_entity),
        'token_receptor': dict(token_receptor),
    }


def _mean_acts(by_entity: dict) -> dict:
    """Convert {entity: [arrays]} → {entity: mean_array}."""
    return {e: np.mean(arrs, axis=0)
            for e, arrs in by_entity.items() if arrs}


def _average_across_seeds(all_mean_acts: list) -> dict:
    """Average a list of {entity: mean_vector} dicts."""
    combined = defaultdict(list)
    for d in all_mean_acts:
        for e, v in d.items():
            combined[e].append(v)
    return {e: np.mean(vecs, axis=0) for e, vecs in combined.items()}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_heatmap(mean_acts: dict, layer_name: str,
                 title: str, save_path: Path):
    entity_types = sorted(e for e in mean_acts if e not in SKIP_ENTITIES)
    if not entity_types:
        return
    matrix  = np.array([mean_acts[e] for e in entity_types])
    n_units = matrix.shape[1]
    step    = max(1, n_units // 32)
    xlabels = [str(i) if i % step == 0 else '' for i in range(n_units)]
    fig, ax = plt.subplots(figsize=(max(14, n_units // 4),
                                    max(4, len(entity_types) + 1)))
    sns.heatmap(matrix, yticklabels=entity_types, xticklabels=xlabels,
                cmap='YlOrRd', cbar_kws={'label': 'Mean Activation'}, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(f'{layer_name} Unit Index', fontsize=11)
    ax.set_ylabel('Entity Type', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      ✓ {save_path.name}")


def compute_rsi(mean_acts: dict) -> list:
    entity_types = sorted(e for e in mean_acts if e not in SKIP_ENTITIES)
    if not entity_types:
        return []
    matrix = np.array([mean_acts[e] for e in entity_types])
    rsi = []
    for u in range(matrix.shape[1]):
        col = matrix[:, u]
        mx, mn = col.max(), col.min()
        rsi.append(float((mx - mn) / mx) if mx > 1e-6 else 0.0)
    return rsi


def plot_rsi_histogram(rsi_scores: list, layer_name: str,
                       title: str, save_path: Path, color: str = 'purple'):
    if not rsi_scores:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rsi_scores, bins=20, color=color, alpha=0.75, edgecolor='black')
    ax.set_xlabel(f'{layer_name} Selectivity Index (RSI)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      ✓ {save_path.name}")


def plot_tsne(mean_acts: dict, n_synthetic: int,
              title: str, save_path: Path, max_per_entity: int = 200,
              raw_lists: dict = None):
    """
    If raw_lists is given (per-token arrays), use them for t-SNE.
    Otherwise fall back to jittered mean vectors.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("      ⚠  scikit-learn not available — skipping t-SNE")
        return

    source = raw_lists if raw_lists else None
    entity_types = sorted(e for e in mean_acts if e not in SKIP_ENTITIES)

    all_vecs, all_labels = [], []
    for entity in entity_types:
        if source and entity in source:
            arrs = source[entity][:max_per_entity]
        else:
            # Jitter the mean vector
            v = mean_acts[entity]
            arrs = [v + np.random.randn(*v.shape) * 0.01
                    for _ in range(min(20, max_per_entity))]
        all_vecs.extend(arrs)
        all_labels.extend([entity] * len(arrs))

    if len(all_vecs) < 5:
        print("      ⚠  Not enough points for t-SNE — skipping")
        return

    X    = np.array(all_vecs)
    perp = min(30, max(5, len(X) // 5))
    X2   = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)

    palette = plt.cm.tab10(np.linspace(0, 0.9, len(entity_types)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, entity in enumerate(entity_types):
        idx = [j for j, lb in enumerate(all_labels) if lb == entity]
        if not idx:
            continue
        ax.scatter(X2[idx, 0], X2[idx, 1], c=[palette[i]], label=entity,
                   alpha=0.65, s=40, edgecolors='none')
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(title='Entity', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      ✓ {save_path.name}")


def save_top_tokens(token_receptor: dict, save_path: Path, top_k: int = 10):
    out = {int(r): [{'token': t, 'activation': a}
                    for t, a in sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]]
           for r, pairs in token_receptor.items()}
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"      ✓ {save_path.name}")


# ---------------------------------------------------------------------------
# Per-seed processing
# ---------------------------------------------------------------------------
def process_one_seed(entry: dict, seed_dir_out: Path,
                     test_loader, vocab_info: dict,
                     use_receptors: bool, use_glomeruli: bool,
                     device, dataset_key: str, exp_type: str) -> dict:
    """
    Run one seed model, save plots in seed_dir_out.
    Returns {'receptor': {entity: mean_vec}, 'glomeruli': {entity: mean_vec}}
    for aggregation at the exp level. Returns None on failure.
    """
    seed_name = entry['seed_dir'].name
    idx2label = vocab_info['idx2label']
    idx2word  = vocab_info.get('idx2word', {})
    num_tags  = len(idx2label)
    f1_tag    = f"F1={entry['test_f1']:.4f}"

    model = load_model(entry, num_tags, device)
    if model is None:
        return None

    acts = collect_activations(model, test_loader, device, idx2label, idx2word)
    del model

    entity_types = sorted(e for e in
                          set(list(acts['receptor'].keys()) +
                              list(acts['glomeruli'].keys()))
                          if e not in SKIP_ENTITIES)

    if not entity_types:
        print(f"      ⚠  No entity activations — skipping plots")
        return None

    seed_dir_out.mkdir(parents=True, exist_ok=True)
    label = f"{dataset_key}/{exp_type}  {seed_name}"

    mean_r = mean_g = None

    # Receptor heatmap + RSI
    if use_receptors and acts['receptor']:
        mean_r = _mean_acts(acts['receptor'])
        plot_heatmap(mean_r, 'Receptor',
                     title=f"Receptor Activations — {label}  ({f1_tag})",
                     save_path=seed_dir_out / 'receptor_heatmap.png')
        rsi_r = compute_rsi(mean_r)
        if rsi_r:
            plot_rsi_histogram(rsi_r, 'Receptor',
                               title=f"Receptor RSI — {label}  ({f1_tag})",
                               save_path=seed_dir_out / 'receptor_rsi.png',
                               color='purple')
    elif not use_receptors:
        print(f"      ↳ use_receptors=False — skipping")

    # Glomeruli heatmap + RSI + t-SNE
    if use_glomeruli and acts['glomeruli']:
        mean_g = _mean_acts(acts['glomeruli'])
        plot_heatmap(mean_g, 'Glomerulus',
                     title=f"Glomeruli Activations — {label}  ({f1_tag})",
                     save_path=seed_dir_out / 'glomeruli_heatmap.png')
        rsi_g = compute_rsi(mean_g)
        if rsi_g:
            plot_rsi_histogram(rsi_g, 'Glomerulus',
                               title=f"Glomeruli RSI — {label}  ({f1_tag})",
                               save_path=seed_dir_out / 'glomeruli_rsi.png',
                               color='orange')
        plot_tsne(mean_g, n_synthetic=0,
                  title=f"t-SNE Glomeruli — {label}  ({f1_tag})",
                  save_path=seed_dir_out / 'tsne.png',
                  raw_lists=acts['glomeruli'])
    elif not use_glomeruli:
        print(f"      ↳ use_glomeruli=False — skipping glomeruli plots")

    # Top tokens per receptor
    if acts['token_receptor']:
        save_top_tokens(acts['token_receptor'],
                        seed_dir_out / 'top_tokens.json')

    return {'receptor': mean_r, 'glomeruli': mean_g}


# ---------------------------------------------------------------------------
# Process one experiment group (best seed only)
# ---------------------------------------------------------------------------
def process_experiment(dataset_key: str, exp_type: str,
                       seed_entries: list, cache_dir: str,
                       batch_size: int, device, output_dir: Path):

    config = seed_entries[0]['config']
    use_receptors = config.get('use_receptors', True)
    use_glomeruli = config.get('use_glomeruli', True)

    if exp_type not in OLFACTORY_EXP_TYPES:
        return

    print(f"\n  [{dataset_key}] {exp_type}  ({len(seed_entries)} seeds)")

    safe_ds = dataset_key.replace('/', '_')
    exp_dir = output_dir / safe_ds / exp_type
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load data ONCE for all seeds in this group
    test_loader, vocab_info = build_test_loader(config, cache_dir, batch_size)
    if test_loader is None:
        return

    # Write best-seed info
    best = seed_entries[0]
    with open(exp_dir / 'best_seed.txt', 'w') as f:
        f.write(f"Best seed : {best['seed_dir'].name}\n"
                f"Test F1   : {best['test_f1']:.4f}\n\n"
                f"All seeds (best→worst):\n")
        for e in seed_entries:
            f.write(f"  {e['seed_dir'].name}  F1={e['test_f1']:.4f}\n")

    # Run only the best seed
    best_name = best['seed_dir'].name
    print(f"    [{best_name}] (Best Seed)  F1={best['test_f1']:.4f}")

    process_one_seed(
        best, exp_dir, test_loader, vocab_info,
        use_receptors, use_glomeruli, device, dataset_key, exp_type,
    )



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Visualize receptor/glomeruli activations — hierarchical output per seed'
    )
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--cache_dir',   type=str, required=True)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--no_cuda',     action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available()
                          else 'cuda')

    print(f"\n{'='*70}")
    print("RECEPTOR / GLOMERULI VISUALIZATIONS — HIERARCHICAL OUTPUT")
    print('='*70)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Cache dir   : {args.cache_dir}")
    print(f"Device      : {device}")
    print('='*70)

    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    print("\nScanning experiments …")
    grouped = find_all_seeds_grouped(results_dir)
    olf     = {k: v for k, v in grouped.items() if k[1] in OLFACTORY_EXP_TYPES}
    print(f"Found {len(olf)} olfactory experiment groups "
          f"({sum(len(v) for v in olf.values())} total seed runs)")

    print('='*70)
    for (dataset_key, exp_type), seed_entries in sorted(olf.items()):
        process_experiment(dataset_key, exp_type, seed_entries,
                           args.cache_dir, args.batch_size, device, output_dir)

    print(f"\n{'='*70}")
    print(f"DONE — outputs saved to: {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
