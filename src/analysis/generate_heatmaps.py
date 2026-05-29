"""
Generate receptor & glomeruli activation heatmaps for olfactory NER models.

For each (dataset, experiment-type) combination only the **best seed** — the one
with the highest test F1 — is used.  Heatmaps are therefore one-per-(dataset,exp)
instead of one per seed run.

Expected directory layout:
    <results_dir>/<dataset_key>/<exp_type>/seed_<N>/
        results.json          ← used to pick best seed
        best_model.pt         ← model weights

Usage:
    python src/analysis/generate_heatmaps.py \
        --results_dir "/content/drive/My Drive/olfaction_inspired_ner/low_resource_exp" \
        --output_dir  "/content/drive/My Drive/olfaction_inspired_ner/analysis_outputs/heatmaps" \
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
# Project root on sys.path so src.* imports work
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent          # …/olfaction-inspired-ner
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Known exp types
# ---------------------------------------------------------------------------
KNOWN_EXP_TYPES = {
    'baseline', 'olfactory', 'receptors_only',
    'no_sparsity', 'more_receptors', 'more_glomeruli',
}


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
    """
    Scan results_dir and return one entry per (dataset_key, exp_type) with
    the seed directory that has the highest test F1.

    Returns list of dicts:
        {
          'dataset':    str,
          'exp_type':   str,
          'seed_dir':   Path,
          'model_path': Path,
          'config':     dict,
          'test_f1':    float,
        }
    """
    # Collect all seed candidates
    # raw[dataset][exp_type] = [(f1, seed_dir_path)]
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

        seed_dir = json_path.parent
        f1 = _test_f1(json_path)
        raw[dataset_key][exp_type].append((f1, seed_dir, json_path))

    best = []
    for dataset_key, exp_map in sorted(raw.items()):
        for exp_type, candidates in sorted(exp_map.items()):
            # Pick seed with highest test F1
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_f1, best_seed_dir, best_json = candidates[0]

            model_path = best_seed_dir / 'best_model.pt'

            # Load config from results.json
            config = {}
            try:
                with open(best_json, 'r') as f:
                    data = json.load(f)
                config = data.get('config', {})
            except Exception:
                pass

            print(f"  [{dataset_key}] {exp_type:20s} "
                  f"best_seed={best_seed_dir.name}  F1={best_f1:.4f}  "
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
# Activation extraction
# ---------------------------------------------------------------------------
def get_activations(model, data_loader, device, idx2label: dict) -> dict:
    """
    Run the model on data_loader and collect mean receptor & glomeruli
    activations per entity type.

    Returns:
        {
          'receptor':  {entity_type: np.ndarray},   # may be empty
          'glomeruli': {entity_type: np.ndarray},   # may be empty
        }
    """
    model.eval()
    receptor_by_entity  = defaultdict(list)
    glomeruli_by_entity = defaultdict(list)

    with torch.no_grad():
        for batch in data_loader:
            sentences, tags, lengths = batch
            sentences = sentences.to(device)
            tags      = tags.to(device)
            lengths   = lengths.to(device)

            try:
                receptors, glomeruli, _ = model.get_receptor_activations(sentences)
            except Exception:
                return {'receptor': {}, 'glomeruli': {}}

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
    """
    Plot a heatmap of mean activations (entities × units).

    activations: {entity_type: mean_vector}
    """
    if not activations:
        print(f"  ⚠  No activations for {layer_name} — skipping.")
        return

    entity_types = sorted(activations.keys())
    matrix = np.array([activations[e] for e in entity_types])  # [E, U]

    # Limit x-axis labels for readability
    n_units = matrix.shape[1]
    if n_units > 64:
        step = max(1, n_units // 32)
        xticklabels = [str(i) if i % step == 0 else '' for i in range(n_units)]
    else:
        xticklabels = list(range(n_units))

    fig, ax = plt.subplots(figsize=(max(14, n_units // 4), max(4, len(entity_types) + 1)))
    sns.heatmap(matrix,
                yticklabels=entity_types,
                xticklabels=xticklabels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Mean Activation'},
                ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f'{layer_name} Unit Index', fontsize=11)
    ax.set_ylabel('Entity Type', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {layer_name} heatmap → {save_path}")


# ---------------------------------------------------------------------------
# Process one experiment (best seed)
# ---------------------------------------------------------------------------
def process_experiment(entry: dict, vocab_info: dict,
                       test_loader, device, output_dir: Path):
    dataset  = entry['dataset']
    exp_type = entry['exp_type']
    config   = entry['config']

    # Only olfactory-type models have receptor/glomeruli layers
    model_type = config.get('model_type', '')
    if model_type not in ('olfactory', 'receptors_only', 'no_sparsity',
                          'more_receptors', 'more_glomeruli'):
        print(f"  ⚠  [{dataset}] {exp_type}: not an olfactory model ({model_type}) — skipping.")
        return

    model_path = entry['model_path']
    if not model_path.exists():
        print(f"  ⚠  [{dataset}] {exp_type}: best_model.pt not found at {model_path} — skipping.")
        return

    print(f"\n  Processing [{dataset}] {exp_type}  (F1={entry['test_f1']:.4f})")

    # --- Load checkpoint ---
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ✗ Cannot load checkpoint: {e}")
        return

    state_dict = ckpt.get('model_state_dict', ckpt)

    # Derive vocab_size and num_tags from the checkpoint
    if 'embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.weight'].shape[0]
    else:
        vocab_size = len(vocab_info.get('word2idx', {}))

    if 'hidden2tag.weight' in state_dict:
        num_tags = state_dict['hidden2tag.weight'].shape[0]
    elif 'output_layer.weight' in state_dict:
        num_tags = state_dict['output_layer.weight'].shape[0]
    else:
        num_tags = len(vocab_info.get('label2idx', {}))

    # --- Build label mapping ---
    if 'label2idx' in ckpt:
        idx2label = {v: k for k, v in ckpt['label2idx'].items()}
    elif 'label2idx' in vocab_info:
        idx2label = {v: k for k, v in vocab_info['label2idx'].items()}
    else:
        idx2label = {i: str(i) for i in range(num_tags)}

    # --- Import model factory ---
    try:
        from src.model.olfactory_ner import create_olfactory_ner
        model = create_olfactory_ner(vocab_size, num_tags, config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  ✗ Cannot instantiate / load model: {e}")
        return

    # --- Extract activations ---
    acts = get_activations(model, test_loader, device, idx2label)

    # --- Save heatmaps ---
    safe_dataset = dataset.replace('/', '_')
    exp_tag = f"{safe_dataset}__{exp_type}"

    for layer in ('receptor', 'glomeruli'):
        if acts[layer]:
            title = (f"{layer.capitalize()} Activations — "
                     f"{dataset} / {exp_type}  (best seed F1={entry['test_f1']:.4f})")
            save_path = output_dir / f'{layer}_heatmap__{exp_tag}.png'
            plot_activation_heatmap(acts[layer], layer.capitalize(), title, save_path)
        else:
            print(f"  ⚠  No {layer} activations extracted for {exp_tag}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate receptor/glomeruli heatmaps using best-seed models'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Root dir of experiment results')
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='Where to save heatmap PNG files')
    parser.add_argument('--data_dir',    type=str, required=True,
                        help='Directory containing raw dataset files (for test_loader)')
    parser.add_argument('--dataset',     type=str, default='conll2003',
                        help='Dataset name to pass to prepare_data (default: conll2003)')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--no_cuda',     action='store_true',
                        help='Force CPU even if CUDA is available')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available()
                          else 'cuda')
    print(f"\n{'='*70}")
    print("HEATMAP GENERATION — BEST SEED PER (DATASET × EXPERIMENT)")
    print('='*70)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Data dir    : {args.data_dir}")
    print(f"Device      : {device}")
    print('='*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("\nLoading dataset for activation extraction …")
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
        print("  Heatmaps require a test loader — aborting.")
        return

    # --- Find best seeds ---
    print("\nFinding best seed per (dataset, experiment) …")
    results_dir = Path(args.results_dir)
    best_entries = find_best_seeds(results_dir)

    if not best_entries:
        print("✗ No results found — nothing to do.")
        return

    olfactory_entries = [e for e in best_entries
                         if e['exp_type'] != 'baseline']
    print(f"\nFound {len(olfactory_entries)} olfactory-type experiment(s) to process.")

    # --- Generate heatmaps ---
    print(f"\n{'='*70}")
    print("GENERATING HEATMAPS")
    print('='*70)

    for entry in olfactory_entries:
        process_experiment(entry, vocab_info, test_loader, device, output_dir)

    print(f"\n{'='*70}")
    print(f"DONE — heatmaps saved to: {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
