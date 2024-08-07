#!/bin/bash
############################################################################################################

python lerobot/scripts/eval.py -p /path/to/pretrained_strong_model eval.n_episodes=500 eval.batch_size=50 eval.use_async_envs=true use_amp=true wandb.enable=true --sampler=bidirectional --reference-policy-name-or-path=/path/to/pretrained_weak_model --ah_test=3  --temperature=0.5 --noise_level=0.0
