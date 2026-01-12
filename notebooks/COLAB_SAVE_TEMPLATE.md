# Add this cell at the END of your Colab notebooks

## 📤 Save Results to GitHub

This cell automatically saves your experiment results and pushes them to GitHub.

```python
# Import helpers
import sys
sys.path.append('/content/olfaction-inspired-ner')

from src.utils.colab_git import save_and_push_experiment

# Save and push results
save_and_push_experiment(
    experiment_name='YOUR_EXPERIMENT_NAME',  # e.g., 'baseline_conll2003', 'olfactory_gelu_ontonotes'
    dataset_name='DATASET_NAME',              # e.g., 'CoNLL-2003', 'OntoNotes5'
    model_type='MODEL_TYPE',                  # e.g., 'baseline', 'olfactory_gelu', 'olfactory_relu'
    config=config,                             # Your config dict
    results=results,                           # Results dict from training
    visualization_dir='./analysis_results',    # Optional: directory with plots
    github_token=None                          # Optional: for private repos
)
```

### For Private Repos

If your repo is private, you need a GitHub token:

1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scopes: `repo` (full control)
4. Copy the token
5. Use it in the cell:

```python
save_and_push_experiment(
    ...
    github_token='ghp_YOUR_TOKEN_HERE'
)
```

⚠️ **Don't share your token!** Delete the cell after pushing.

---

## What Gets Saved?

For each experiment:
- ✅ `metadata.json` - Full experiment configuration
- ✅ `results.json` - All metrics (train/valid/test, per-entity)
- ✅ `SUMMARY.md` - Human-readable summary
- ✅ `visualizations/` - All plots, heatmaps, t-SNE (if provided)

Plus:
- ✅ Auto-generated index in `experiment_results/README.md`
- ✅ Easy comparison table on GitHub

---

## Example Usage

```python
# At the end of your training script
save_and_push_experiment(
    experiment_name='olfactory_gelu_conll2003',
    dataset_name='CoNLL-2003',
    model_type='olfactory_gelu',
    config={
        'embed_dim': 300,
        'num_receptors': 128,
        'num_glomeruli': 32,
        'receptor_activation': 'gelu',
        'lstm_hidden': 256,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'batch_size': 32
    },
    results={
        'test': {
            'f1': 0.7306,
            'precision': 0.7873,
            'recall': 0.6815,
            'per_entity': {
                'LOC': 0.8082,
                'MISC': 0.6761,
                'ORG': 0.7067,
                'PER': 0.6914
            }
        }
    },
    visualization_dir='./analysis_gelu'
)
```

✅ Results pushed! View at:
https://github.com/YOUR_USERNAME/olfaction-inspired-ner/tree/main/experiment_results
