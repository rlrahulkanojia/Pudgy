#!/bin/bash
# Precompute VAE latents + T5 embeddings for the v8 joint set (idempotent: --skip_existing).
set -euo pipefail
source /workspace/Pudgy/.venv-wan/bin/activate
cd /workspace/musubi-tuner
CFG=/workspace/data_v8/dataset_config_joint_v8.workspace.toml
M=/workspace/wan_models
python src/musubi_tuner/wan_cache_latents.py --dataset_config $CFG \
  --vae $M/comfy21/split_files/vae/wan_2.1_vae.safetensors --i2v --skip_existing --num_workers 16
python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config $CFG \
  --t5 $M/t5/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16 --skip_existing --num_workers 16
du -sh /workspace/wan_cache/latents_v8/joint
find /workspace/wan_cache/latents_v8/joint -name "*.safetensors" | wc -l
