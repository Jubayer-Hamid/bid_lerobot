#!/bin/bash
############################################################################################################

python lerobot/scripts/eval.py -p /iris/u/jubayer/lerobot/outputs/train/2024-07-18/12-28-37_pusht_vqbet_default/checkpoints/200000/pretrained_model eval.n_episodes=500 eval.batch_size=50 eval.use_async_envs=true use_amp=true wandb.enable=true --sampler=random --ah_test=1 --temperature=1.0