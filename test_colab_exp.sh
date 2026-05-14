#!/bin/bash
# Colab Test Script for Olfaction-Inspired NER
# Runs a quick 1-epoch test sequentially across all datasets and experiments for a single seed to verify setup.

BASE_SAVE_DIR="/content/drive/My Drive/olfaction_inspired_ner_test"
SEED=42
EXPERIMENTS=("baseline" "olfactory")
DATASETS=("conll_en" "wikiann_mr" "wikiann_hi" "wikiann_ta" "wikiann_bn" "wikiann_te")
FASTTEXT_LANGS=("en" "mr" "hi" "ta" "bn" "te")

echo "========================================"
echo "Preparing FastText Embeddings (Testing)"
echo "========================================"
mkdir -p data
for lang in "${FASTTEXT_LANGS[@]}"; do
    if [ ! -f "data/cc.${lang}.300.vec" ]; then
        echo "Downloading FastText for ${lang}..."
        wget -nc -P data/ "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lang}.300.vec.gz"
        gunzip "data/cc.${lang}.300.vec.gz"
    else
        echo "FastText for ${lang} already exists."
    fi
done

echo "========================================"
echo "Starting 1-Epoch Tests"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Testing Dataset: ${dataset}"
    echo "----------------------------------------"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "Testing ${exp} on ${dataset} (1 Epoch)..."
        python src/train_universal.py \
            --config config/universal_config.yaml \
            --dataset_key ${dataset} \
            --experiment ${exp} \
            --save_dir "${BASE_SAVE_DIR}" \
            --seed ${SEED} \
            --epochs 1
    done
done

echo "All tests completed!"
