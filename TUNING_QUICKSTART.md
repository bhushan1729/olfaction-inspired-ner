# Quick Start: Tuning Experiments

## Run a Single Experiment

```bash
# GELU activation
python src/train.py --config config/tuning_experiments.yaml \
  --experiment activation_gelu \
  --save_dir results/tuning/gelu

# Swish activation  
python src/train.py --config config/tuning_experiments.yaml \
  --experiment activation_swish \
  --save_dir results/tuning/swish

# 256 receptors
python src/train.py --config config/tuning_experiments.yaml \
  --experiment more_receptors \
  --save_dir results/tuning/receptors_256

# Strong diversity loss
python src/train.py --config config/tuning_experiments.yaml \
  --experiment strong_diversity \
  --save_dir results/tuning/diversity_0.05

# Best combo: GELU + more receptors + strong diversity
python src/train.py --config config/tuning_experiments.yaml \
  --experiment gelu_more_receptors \
  --save_dir results/tuning/gelu_256

# Minimal model (best from previous): GELU + no glomeruli
python src/train.py --config config/tuning_experiments.yaml \
  --experiment minimal_gelu \
  --save_dir results/tuning/minimal_gelu
```

## Recommended Experiment Order

1. **Start with GELU** (easiest, most promising):
   ```bash
   python src/train.py --config config/tuning_experiments.yaml \
     --experiment activation_gelu --save_dir results/tuning/gelu
   ```

2. **Then minimal_gelu** (best architecture + best activation):
   ```bash
   python src/train.py --config config/tuning_experiments.yaml \
     --experiment minimal_gelu --save_dir results/tuning/minimal_gelu
   ```

3. **If results good, try gelu_more_receptors**:
   ```bash
   python src/train.py --config config/tuning_experiments.yaml \
     --experiment gelu_more_receptors --save_dir results/tuning/gelu_256
   ```

## Expected Results

- **GELU**: May improve F1 by 0.5-1% over ReLU
- **Minimal GELU**: Could beat baseline if GELU helps
- **More receptors**: Better interpretability, similar F1
- **Strong diversity**: More specialized receptors (visible in heatmaps)

## Time Estimates

- Each experiment: ~20-30 minutes on Colab GPU
- Recommended 3 experiments: ~1-1.5 hours total
