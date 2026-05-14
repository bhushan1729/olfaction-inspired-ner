#!/bin/bash
# Colab Run Script for Olfaction-Inspired NER (Parallel Seeds)
# Runs seeds in parallel to utilize more of the available 15GB GPU VRAM.

BASE_SAVE_DIR="/content/drive/My Drive/olfaction_inspired_ner"
SEEDS=(42 123 456 789 1011)
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
        gunzip "data/cc.${lang}.300.vec.gz"
    else
        echo "FastText for ${lang} already exists."
    fi
done

echo "========================================"
echo "Starting Parallel Experiments"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Running Dataset: ${dataset}"
    echo "----------------------------------------"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "Launching 5 seeds in parallel for ${exp} on ${dataset}..."
        for seed in "${SEEDS[@]}"; do
            python src/train_universal.py \
                --config config/universal_config.yaml \
                --dataset_key ${dataset} \
                --experiment ${exp} \
                --save_dir "${BASE_SAVE_DIR}" \
                --seed ${seed} > /dev/null 2>&1 &
        done
        # Wait for all 5 seeds of this experiment to finish before moving to the next
        wait
        echo "Finished ${exp} on ${dataset} for all seeds."
    done
done

echo "All parallel experiments completed!"
