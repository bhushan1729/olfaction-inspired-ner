"""
Generate receptor & glomeruli activation heatmaps for olfactory NER models.

Output hierarchy:
    <output_dir>/
      <dataset_key>/
        <exp_type>/
          seed_<N>/
            receptor_heatmap.png        ← per-seed
            glomeruli_heatmap.png       ← per-seed (if use_glomeruli)
          receptor_heatmap_all_seeds.png  ← mean across ALL seeds
          glomeruli_heatmap_all_seeds.png
          best_seed.txt
        ...

For each experiment group (dataset, exp_type) the test data is loaded ONCE
(matching the model's own dataset/language), then every seed model is run in turn.

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
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOWN_EXP_TYPES = {
    'baseline', 'olfactory', 'receptors_only',
    'no_sparsity', 'more_receptors', 'more_glomeruli',
}
OLFACTORY_EXP_TYPES = KNOWN_EXP_TYPES - {'baseline'}


# ---------------------------------------------------------------------------
# Scanning — group ALL seeds by (dataset, exp_type)
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
    Returns {
        (dataset_key, exp_type): [
            {'seed_dir': Path, 'model_path': Path, 'config': dict, 'test_f1': float},
            ...                                                     (sorted best→worst)
        ]
    }
    """
    raw = defaultdict(list)

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

        config = {}
        try:
            with open(json_path) as f:
                config = json.load(f).get('config', {})
        except Exception:
            pass

        model_path = json_path.parent / 'best_model.pt'
        raw[(dataset_key, exp_type)].append({
            'seed_dir':   json_path.parent,
            'model_path': model_path,
            'config':     config,
            'test_f1':    _test_f1(json_path),
        })

    # Sort seeds best→worst within each group
    grouped = {}
    for key, entries in raw.items():
        entries.sort(key=lambda e: e['test_f1'], reverse=True)
        grouped[key] = entries

    return grouped


# ---------------------------------------------------------------------------
# Data loader for an experiment (built from its own dataset/language)
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
        print(f"    ✓ {len(test_loader.dataset)} test samples  "
              f"labels={list(vocab_info['idx2label'].values())}")
        return test_loader, vocab_info['idx2label']
    except Exception as e:
        print(f"    ✗ Data load failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(entry: dict, num_tags: int, device):
    """Load and return the model for one seed, or None on failure."""
    from src.model.olfactory_ner import create_olfactory_ner
    model_path = entry['model_path']
    config     = entry['config']

    if not model_path.exists():
        print(f"      ✗ best_model.pt not found: {model_path}")
        return None

    try:
        ckpt       = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        vocab_size = (state_dict['embedding.weight'].shape[0]
                      if 'embedding.weight' in state_dict else 0)
        model = create_olfactory_ner(vocab_size, num_tags, config)
        model.load_state_dict(state_dict, strict=False)
        return model.to(device).eval()
    except Exception as e:
        print(f"      ✗ Model load error: {e}")
        return None


# ---------------------------------------------------------------------------
# Activation extraction — returns {entity: mean_vector}
# ---------------------------------------------------------------------------
def get_activations(model, data_loader, device, idx2label: dict) -> dict:
    """Returns {'receptor': {entity: ndarray}, 'glomeruli': {entity: ndarray}}."""
    receptor_by_entity  = defaultdict(list)
    glomeruli_by_entity = defaultdict(list)
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
                        receptor_by_entity[entity].append(receptors[i, t].cpu().numpy())
                    if glomeruli is not None:
                        glomeruli_by_entity[entity].append(glomeruli[i, t].cpu().numpy())

    if first_exc and n_good == 0:
        print(f"      ✗ get_receptor_activations failed: {first_exc}")

    def _mean(d):
        return {e: np.mean(acts, axis=0) for e, acts in d.items() if acts}

    return {'receptor': _mean(receptor_by_entity), 'glomeruli': _mean(glomeruli_by_entity)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
SKIP_ENTITIES = {'micro avg', 'macro avg', 'weighted avg', 'O'}


def plot_heatmap(acts: dict, layer_name: str, title: str, save_path: Path):
    """acts: {entity: mean_vector}"""
    entity_types = sorted(e for e in acts if e not in SKIP_ENTITIES)
    if not entity_types:
        return
    matrix = np.array([acts[e] for e in entity_types])
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


def _average_acts(all_acts: list) -> dict:
    """Average a list of {entity: mean_vector} dicts."""
    combined = defaultdict(list)
    for acts in all_acts:
        for entity, vec in acts.items():
            combined[entity].append(vec)
    return {e: np.mean(vecs, axis=0) for e, vecs in combined.items()}


# ---------------------------------------------------------------------------
# Process one experiment group (best seed only)
# ---------------------------------------------------------------------------
def process_experiment(dataset_key: str, exp_type: str,
                       seed_entries: list, cache_dir: str,
                       batch_size: int, device, output_dir: Path):

    config = seed_entries[0]['config']
    model_type    = config.get('model_type', exp_type)
    use_receptors = config.get('use_receptors', True)
    use_glomeruli = config.get('use_glomeruli', True)

    if exp_type not in OLFACTORY_EXP_TYPES:
        return

    print(f"\n  [{dataset_key}] {exp_type}  ({len(seed_entries)} seeds)")

    # Exp-level output dir
    safe_ds   = dataset_key.replace('/', '_')
    exp_dir   = output_dir / safe_ds / exp_type
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load test data ONCE for this exp (all seeds share same dataset/language)
    test_loader, idx2label = build_test_loader(config, cache_dir, batch_size)
    if test_loader is None:
        return

    num_tags = len(idx2label)

    # Save best-seed info
    best = seed_entries[0]  # already sorted best→worst
    with open(exp_dir / 'best_seed.txt', 'w') as f:
        f.write(f"Best seed : {best['seed_dir'].name}\n"
                f"Test F1   : {best['test_f1']:.4f}\n\n"
                f"All seeds (best→worst):\n")
        for e in seed_entries:
            f.write(f"  {e['seed_dir'].name}  F1={e['test_f1']:.4f}\n")

    # Run only the best seed
    best_name = best['seed_dir'].name
    print(f"    [{best_name}] (Best Seed)  F1={best['test_f1']:.4f}")

    model = load_model(best, num_tags, device)
    if model is None:
        return

    acts = get_activations(model, test_loader, device, idx2label)
    del model  # free memory

    # Save heatmaps directly in exp_dir
    if use_receptors and acts['receptor']:
        plot_heatmap(
            acts['receptor'], 'Receptor',
            title=f"Receptor Activations — {dataset_key}/{exp_type}  {best_name}",
            save_path=exp_dir / 'receptor_heatmap.png',
        )
    elif not use_receptors:
        print(f"      ↳ use_receptors=False — skipping receptor heatmap")

    if use_glomeruli and acts['glomeruli']:
        plot_heatmap(
            acts['glomeruli'], 'Glomerulus',
            title=f"Glomeruli Activations — {dataset_key}/{exp_type}  {best_name}",
            save_path=exp_dir / 'glomeruli_heatmap.png',
        )
    elif not use_glomeruli:
        print(f"      ↳ use_glomeruli=False — skipping glomeruli heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate receptor/glomeruli heatmaps — hierarchical output per seed'
    )
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--cache_dir',   type=str, required=True,
                        help='Dataset cache directory')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--no_cuda',     action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available()
                          else 'cuda')

    print(f"\n{'='*70}")
    print("HEATMAP GENERATION — HIERARCHICAL OUTPUT")
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

    olf_groups = {k: v for k, v in grouped.items()
                  if k[1] in OLFACTORY_EXP_TYPES}

    print(f"\nFound {len(olf_groups)} olfactory experiment groups "
          f"({sum(len(v) for v in olf_groups.values())} total seed runs)\n")

    print('='*70)
    for (dataset_key, exp_type), seed_entries in sorted(olf_groups.items()):
        process_experiment(dataset_key, exp_type, seed_entries,
                           args.cache_dir, args.batch_size, device, output_dir)

    print(f"\n{'='*70}")
    print(f"DONE — outputs saved to: {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
