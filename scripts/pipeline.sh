#!/usr/bin/env bash

# set -e

# PYTHON=python3
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CONFIG="configs/config.yaml"           
# INPUT="inputs/musique_s.json"       
# OUTPUT="results/pipeline_results"   

mkdir -p "results/pipeline_v5_results_new/musique"

python3 src/pipeline_v5.py \
    --input data/infinitebench_musique/passkey_musique_annotated.json \
    --output_dir results/pipeline_v5_results_new/passkey/full \
    --model_id mistralai/Mistral-7B-Instruct-v0.2 \
    --top_k 20 \
    --max_tokens 10 \
    --device cuda:0 \
    --sparsity_ratio 1.0

