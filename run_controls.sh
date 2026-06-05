#!/bin/bash
# Colab Run Script for Olfaction-Inspired NER - Control Baselines
# Runs sequentially. (Parallel execution causes CPU/Disk bottlenecks on Colab).

BASE_SAVE_DIR="/content/drive/My Drive/olfaction_inspired_ner/control_baseline_exp"
SEEDS=(42 123 456 789 1011)
EXPERIMENTS=("dense_bottleneck" "simple_sparse_bottleneck" "sparse_bottleneck_l1")
DATASETS=("conll_en_1k" "wikiann_mr_1k" "wikiann_hi_1k" "wikiann_bn_1k" "wikiann_ta_1k" "wikiann_te_1k")

# Ensure UTF-8 output encoding to prevent Windows/Colab checkmark print crashes
export PYTHONIOENCODING=utf-8

echo "========================================"
echo "Starting Sequential Control Experiments"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Running Dataset: ${dataset}"
    echo "----------------------------------------"
    for exp in "${EXPERIMENTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running ${exp} on ${dataset} with seed ${seed}..."
            python src/train_universal.py \
                --config config/universal_config.yaml \
                --dataset_key ${dataset} \
                --experiment ${exp} \
                --save_dir "${BASE_SAVE_DIR}" \
                --seed ${seed}
        done
    done
done

echo "All control baseline experiments completed!"
