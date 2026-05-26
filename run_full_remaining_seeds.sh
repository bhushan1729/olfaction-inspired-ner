#!/bin/bash
# Script to run the remaining seeds (789 and 1011) for full-scale experiments sequentially.
# This completes the 5-seed setup across all models and datasets.

BASE_SAVE_DIR="/content/drive/My Drive/olfaction_inspired_ner/no_pretrained_embeddings"
SEEDS=(789 1011)
EXPERIMENTS=("baseline" "olfactory" "receptors_only" "no_sparsity" "more_receptors" "more_glomeruli")
DATASETS=("conll_en" "wikiann_mr" "wikiann_hi" "wikiann_ta" "wikiann_bn" "wikiann_te")
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
echo "Starting Sequential Full-Scale Remaining Seeds Experiments"
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

echo "Remaining full-scale seed experiments completed!"
