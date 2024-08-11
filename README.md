# Bidirectional Behavioral Sampling

This repository accompanies the paper **Bidirectional Action Decoding: Understanding and Improving Action Chunking for Generative Behavior Cloning**. 

We adapted the [LeRobot repository](https://github.com/huggingface/lerobot) to implement Bidirectional Behavioral Sampling on the VQ-BeT policy. Our experiments run VQ-BeT on the PushT environment. 

Follow the setup instructions provided in the [LeRobot repository](https://github.com/huggingface/lerobot). 

## Evaluating pretrained policies
This repository allows one to evaluate a pretrained policy using (1) random sampling (2) backward decision coherence sampling (3) forward contrastive sampling and (4) bidirectional behavioral sampling. 

Our experiments uses the [LeRobot pre-trained checkpoint](https://huggingface.co/lerobot/vqbet_pusht). To download the checkpoint, first run ```pip install huggingface_hub```. Then, run: 

```
from pathlib import Path
from huggingface_hub import snapshot_download

# Define the path where you want to download the checkpoint
download_path = Path("/path/to/save/checkpoint/vqbet_pusht")

# Download the checkpoint directory
snapshot_download(repo_id="lerobot/vqbet_pusht", local_dir=download_path)
```

Forward contrastive sampling and bidirectional behavioral sampling require a reference policy. In our experiments, we used an [early checkpoint](https://drive.google.com/drive/u/0/folders/1FXHzPZPfTO7SCM-OTKUy3EEvKKTq3LC4) for the source of negative samples. One can direclty download the ["pretrained_weak_model"](https://drive.google.com/drive/u/0/folders/1kBPDBcPU3gLYCZNxkoRpXQ_e3tqi9lPR) and use it as the reference policy.. 


To use random sampling, run the following command:
```
bash commands/eval_random_sampler.sh
```
This script requires: 
(1) a pre-trained policy. We use the [LeRobot pre-trained checkpoint](https://huggingface.co/lerobot/vqbet_pusht). 
(2) specify the action horizon to be used. For example, for closed-loop sampling, we use ```--ah_test=1```. For open-loop sampling, we use ```aH_test=5```. 
(3) the temperature to be used in the softmax layer. For example, ```--temperature=0.5``` 
(4) the noise level in the environment. For example, for a deterministic environment we use ```--noise_level=0.0```.

To use backward decision coherence sampling, run the following command:
```
bash commands/eval_coherence_sampler.sh
```
This requires the same arguments as random sampling. 

To use forward contrastive sampling, run the following command:
```
bash commands/eval_contrastive_sampler.sh
```
This requires the following additional arguments:
(1) a weak policy for contrastive sampling. We use the [early checkpoint](https://drive.google.com/drive/u/0/folders/1FXHzPZPfTO7SCM-OTKUy3EEvKKTq3LC4). 


To use bidirectional action decoding, run the following command:
```
bash commands/eval_bidirectional_sampler.sh
```
This requires the same arguments as contrastive sampling. 

## Expected results 

We provide expected success rate for comparison of vanilla sampling and BBS (both open and closed loop):

|              | Vanilla Open Loop | BBS Open Loop | Vanilla Closed Loop | BBS Closed Loop |
|--------------|-------------------|---------------|---------------------|-----------------|
| Noise = 0.0  | 61.0              | **65.2**      | 52.0                | **56.6**        |
| Noise = 1.0  | 39.0              | 39.8          | 50.4                | **54.8**        |
| Noise = 1.5  | 19.4              | 21.4          | 44.2                | **54.4**        |

For vanilla sampling, we use the default LeRobot ```temperature=0.1```. For BBS, we used ```temperature=0.5``` to ensure sampling diversity. 

Furthermore, to support our theoretical analysis, we evaluate the performance of various action horizons in clean and noisy environments. The expected success rates are:

| Horizon          |   1     |   3  |   5  |
|------------------|---------|------|------|
| Noise = 0.0      |  52.0   | 57.6 | 61.0 |
| Noise = 1.5      |  44.2   | 43.4 | 19.4 |
