#!/bin/bash
############################################################################################################

python lerobot/scripts/eval.py -p /iris/u/jubayer/lerobot/vqbet_pusht eval.n_episodes=500 eval.batch_size=50 eval.use_async_envs=true use_amp=true wandb.enable=true --sampler=contrastive --ah_test=1 --reference-policy-name-or-path=/iris/u/jubayer/lerobot/outputs/train/2024-07-23/21-44-50_pusht_vqbet_default/checkpoints/100000/pretrained_model --temperature=0.5
