#!/bin/bash
############################################################################################################

python lerobot/scripts/eval.py -p /lerobot/ckpt/pretrained_strong_model eval.n_episodes=500 eval.batch_size=50 eval.use_async_envs=true use_amp=true wandb.enable=true --sampler=contrastive --ah_test=1 --reference-policy-name-or-path=/lerobot/ckpt/pretrained_weak_model --temperature=0.5
