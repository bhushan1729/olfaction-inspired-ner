#!/bin/bash
# Script to run the remaining 1k-capped experiments sequentially.
# Targets the configurations that were not previously completed.

BASE_SAVE_DIR="/content/drive/My Drive/olfaction_inspired_ner/low_resource_exp"
SEEDS=(42 123 456 789 1011)
EXPERIMENTS=("receptors_only" "more_receptors" "more_glomeruli")
DATASETS=("conll_en_1k" "wikiann_mr_1k" "wikiann_hi_1k" "wikiann_bn_1k" "wikiann_ta_1k" "wikiann_te_1k")
FASTTEXT_LANGS=("en" "mr" "hi" "ta" "bn" "te")

echo "========================================"
echo "Preparing FastText Embeddings"
echo "========================================"
mkdir -p data
for lang in "${FASTTEXT_LANGS[@]}"; do
    if [ ! -f "data/cc.${lang}.300.vec" ]; then
        echo "Downloading FastText for ${lang}..."
        wget -nc -P data/ "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lang}.300.vec.gz"
        gunzip -f "data/cc.${lang}.300.vec.gz"
    else
        echo "FastText for ${lang} already exists."
    fi
done

echo "========================================"
echo "Starting Sequential Remaining Low-Resource Experiments"
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

echo "Remaining low-resource experiments completed!"
