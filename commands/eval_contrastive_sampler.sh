#!/bin/bash
############################################################################################################

python lerobot/scripts/eval.py -p /iris/u/jubayer/lerobot/outputs/train/2024-07-18/12-28-37_pusht_vqbet_default/checkpoints/200000/pretrained_model eval.n_episodes=500 eval.batch_size=50 eval.use_async_envs=true use_amp=true wandb.enable=true --sampler=contrastive --ah_test=1 --reference-policy-name-or-path=/iris/u/jubayer/lerobot/outputs/train/2024-07-18/12-28-37_pusht_vqbet_default/checkpoints/100000/pretrained_model --temperature=1.0 --noise_level=0.0