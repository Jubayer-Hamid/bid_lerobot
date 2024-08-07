#!/bin/bash
############################################################################################################

python lerobot/scripts/eval.py -p /iris/u/jubayer/lerobot/vqbet_pusht eval.n_episodes=500 eval.batch_size=50 eval.use_async_envs=true use_amp=true wandb.enable=true --sampler=coherence --ah_test=1 --temperature=0.5
