"""
Generate receptor & glomeruli activation heatmaps for olfactory NER models.

For each (dataset, experiment-type) combination only the **best seed** — the one
with the highest test F1 — is used.

Key fix: each model is evaluated on its OWN test split (re-loaded using the
dataset / language from its results.json config), so WikiANN models are not
mistakenly evaluated on CoNLL data.

Expected directory layout:
    <results_dir>/<dataset_key>/<exp_type>/seed_<N>/
        results.json          ← used to pick best seed + read config
        best_model.pt         ← model weights

Usage:
    python src/analysis/generate_heatmaps.py \
        --results_dir "/content/drive/My Drive/olfaction_inspired_ner/low_resource_exp" \
        --output_dir  "/content/drive/My Drive/olfaction_inspired_ner/analysis_outputs/heatmaps" \
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
# Project root on sys.path so src.* imports work
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
# Helpers
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
    """Return one entry per (dataset_key, exp_type) — best test-F1 seed."""
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
# Build the correct test loader for a given experiment config
# ---------------------------------------------------------------------------
def build_test_loader(config: dict, cache_dir: str, batch_size: int = 32):
    """
    Load the test data that matches this experiment's training data.

    Uses the 'dataset' and 'language' fields from the results.json config.
    Returns (test_loader, idx2label) or (None, None) on failure.
    """
    from src.data.unified_loader import get_dataset

    dataset_name = config.get('dataset', 'conll2003')
    language     = config.get('language', None)
    min_freq     = config.get('min_freq', 2)
    max_train    = config.get('max_train_samples', None)

    print(f"    Loading data: dataset={dataset_name}  language={language}")
    try:
        _, _, test_loader, vocab_info = get_dataset(
            dataset_name=dataset_name,
            language=language,
            cache_dir=cache_dir,
            batch_size=batch_size,
            min_freq=min_freq,
            max_train_samples=max_train,
        )
        idx2label = vocab_info['idx2label']
        print(f"    ✓ Test loader: {len(test_loader.dataset)} samples  "
              f"|  labels={list(idx2label.values())}")
        return test_loader, idx2label
    except Exception as e:
        print(f"    ✗ Failed to load data: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
def get_activations(model, data_loader, device, idx2label: dict) -> dict:
    """
    Run model on data_loader; collect mean receptor & glomeruli activations
    per entity type.

    Returns {'receptor': {entity: ndarray}, 'glomeruli': {entity: ndarray}}
    """
    model.eval()
    receptor_by_entity  = defaultdict(list)
    glomeruli_by_entity = defaultdict(list)

    first_exception = None
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            sentences, tags, lengths = batch
            sentences = sentences.to(device)
            tags      = tags.to(device)
            lengths   = lengths.to(device)

            try:
                receptors, glomeruli, _ = model.get_receptor_activations(sentences)
            except Exception as e:
                if first_exception is None:
                    first_exception = e
                continue  # skip bad batch, don't abort

            n_batches += 1

            for i, length in enumerate(lengths):
                L = int(length.item())
                for t in range(L):
                    label = idx2label.get(int(tags[i, t].item()), 'O')
                    if label == 'O':
                        continue
                    entity_type = label.split('-', 1)[-1]

                    if receptors is not None:
                        receptor_by_entity[entity_type].append(
                            receptors[i, t].cpu().numpy())
                    if glomeruli is not None:
                        glomeruli_by_entity[entity_type].append(
                            glomeruli[i, t].cpu().numpy())

    if first_exception and n_batches == 0:
        print(f"    ✗ get_receptor_activations failed on every batch: {first_exception}")

    def _mean(d):
        return {e: np.mean(acts, axis=0) for e, acts in d.items() if acts}

    return {
        'receptor':  _mean(receptor_by_entity),
        'glomeruli': _mean(glomeruli_by_entity),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_activation_heatmap(activations: dict, layer_name: str,
                            title: str, save_path: Path):
    if not activations:
        print(f"    ⚠  No {layer_name} activations — skipping heatmap.")
        return

    entity_types = sorted(activations.keys())
    matrix = np.array([activations[e] for e in entity_types])

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
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f'{layer_name} Unit Index', fontsize=11)
    ax.set_ylabel('Entity Type', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {layer_name} heatmap → {save_path.name}")


# ---------------------------------------------------------------------------
# Process one experiment (best seed)
# ---------------------------------------------------------------------------
def process_experiment(entry: dict, cache_dir: str,
                       batch_size: int, device, output_dir: Path):
    dataset  = entry['dataset']
    exp_type = entry['exp_type']
    config   = entry['config']
    model_path = entry['model_path']

    model_type = config.get('model_type', exp_type)
    if model_type not in OLFACTORY_EXP_TYPES and exp_type not in OLFACTORY_EXP_TYPES:
        print(f"  ⚠  [{dataset}] {exp_type}: not an olfactory model — skipping.")
        return

    if not model_path.exists():
        print(f"  ⚠  [{dataset}] {exp_type}: best_model.pt not found — skipping.")
        return

    print(f"\n  Processing [{dataset}] {exp_type}  "
          f"(F1={entry['test_f1']:.4f}  seed={entry['seed_dir'].name})")

    # --- Load the correct test data for this model ---
    test_loader, idx2label = build_test_loader(config, cache_dir, batch_size)
    if test_loader is None:
        return

    # --- Load checkpoint ---
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"    ✗ Cannot load checkpoint: {e}")
        return

    state_dict = ckpt.get('model_state_dict', ckpt)

    vocab_size = (state_dict['embedding.weight'].shape[0]
                  if 'embedding.weight' in state_dict else 0)
    if 'hidden2tag.weight' in state_dict:
        num_tags = state_dict['hidden2tag.weight'].shape[0]
    elif 'output_layer.weight' in state_dict:
        num_tags = state_dict['output_layer.weight'].shape[0]
    else:
        num_tags = len(idx2label)

    # --- Build & load model ---
    try:
        from src.model.olfactory_ner import create_olfactory_ner
        model = create_olfactory_ner(vocab_size, num_tags, config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"    ✗ Cannot instantiate model: {e}")
        return

    # --- Extract activations ---
    print("    Collecting activations …")
    acts = get_activations(model, test_loader, device, idx2label)

    skip_entities = {'micro avg', 'macro avg', 'weighted avg', 'O'}
    entity_types  = sorted(e for e in
                           set(list(acts['receptor'].keys()) +
                               list(acts['glomeruli'].keys()))
                           if e not in skip_entities)

    if not entity_types:
        print("    ⚠  No entity activations collected — skipping.")
        return

    # --- Save heatmaps ---
    safe_ds = dataset.replace('/', '_')
    prefix  = f"{safe_ds}__{exp_type}"
    f1_tag  = f"F1={entry['test_f1']:.4f}"

    for layer in ('receptor', 'glomeruli'):
        valid = [e for e in entity_types if e in acts[layer]]
        if not valid:
            continue
        matrix = np.array([acts[layer][e] for e in valid])
        title  = (f"{layer.capitalize()} Activations — "
                  f"{dataset} / {exp_type}  ({f1_tag})")
        save_path = output_dir / f'{prefix}__{layer}_heatmap.png'
        plot_activation_heatmap(acts[layer], layer.capitalize(), title, save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate receptor/glomeruli heatmaps (best-seed per experiment)'
    )
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--cache_dir',   type=str, required=True,
                        help='Directory to cache/store dataset files')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--no_cuda',     action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available()
                          else 'cuda')

    print(f"\n{'='*70}")
    print("HEATMAP GENERATION — BEST SEED PER (DATASET × EXPERIMENT)")
    print('='*70)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Cache dir   : {args.cache_dir}")
    print(f"Device      : {device}")
    print('='*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nFinding best seed per (dataset, experiment) …")
    results_dir  = Path(args.results_dir)
    best_entries = find_best_seeds(results_dir)

    olf_entries = [e for e in best_entries if e['exp_type'] in OLFACTORY_EXP_TYPES]
    print(f"\nFound {len(olf_entries)} olfactory-type experiment(s) to process.")

    print(f"\n{'='*70}")
    print("GENERATING HEATMAPS")
    print('='*70)

    for entry in olf_entries:
        process_experiment(entry, args.cache_dir, args.batch_size, device, output_dir)

    print(f"\n{'='*70}")
    print(f"DONE — heatmaps saved to: {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
