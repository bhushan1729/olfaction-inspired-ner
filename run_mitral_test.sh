#!/bin/bash

# run_mitral_test.sh
# Tests baseline and olfactory_mitral_base on all datasets for 1 seed

CONFIG_PATH="config/mitral_config.yaml"
BASE_SAVE_DIR="/content/drive/My Drive/olfaction_inspired_ner"
SEED=42
EPOCHS=30

echo "========================================"
echo "Starting Mitral Hypothesis Tests"
echo "========================================"

# Define languages and datasets
declare -A datasets
datasets["conll_en"]="en"
datasets["wikiann_mr"]="mr"
datasets["wikiann_hi"]="hi"
datasets["wikiann_ta"]="ta"
datasets["wikiann_bn"]="bn"
datasets["wikiann_te"]="te"

# Define experiments to run
experiments=("baseline" "olfactory_mitral_base")

# Iterate over each dataset
for ds in "${!datasets[@]}"; do
    lang=${datasets[$ds]}
    
    echo "----------------------------------------"
    echo "Running Dataset: $ds"
    echo "----------------------------------------"
    
    for exp in "${experiments[@]}"; do
        echo "Running $exp on $ds with seed $SEED..."
        
        python src/train_universal.py \
            --config "$CONFIG_PATH" \
            --dataset_key "$ds" \
            --experiment "$exp" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --save_dir "$BASE_SAVE_DIR"
            
        if [ $? -ne 0 ]; then
            echo "Error running $exp on $ds. Exiting."
            exit 1
        fi
    done
done

echo "========================================"
echo "All Mitral Tests Completed Successfully!"
echo "========================================"
