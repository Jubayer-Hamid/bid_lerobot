# Bidirectional Decoding

This repository accompanies the paper **Bidirectional Decoding: Improving Action Chunking via Closed-Loop Resampling** by Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du and Chelsea Finn. 

We adapted the [LeRobot repository](https://github.com/huggingface/lerobot) to implement Bidirectional Decoding on the VQ-BeT policy. Our experiments run VQ-BeT on the PushT environment. 

Follow the setup instructions provided in the [LeRobot repository](https://github.com/huggingface/lerobot). 

## Evaluating pretrained policies
This repository allows one to evaluate a pretrained policy using (1) random sampling (2) backward coherence sampling (3) forward contrast sampling and (4) bidirectional decoding. 

Our experiments uses the [LeRobot pre-trained checkpoint](https://huggingface.co/lerobot/vqbet_pusht). To download the checkpoint, first run ```pip install huggingface_hub```. Then, create a directory to save the pre-trained checkpoints:

```
mkdir -p lerobot/ckpt
```

Then, run: 

```
from pathlib import Path
from huggingface_hub import snapshot_download

# Define the path where you want to download the checkpoint
download_path = Path("/lerobot/ckpt")

# Download the checkpoint directory
snapshot_download(repo_id="lerobot/vqbet_pusht", local_dir=download_path)
```

Forward contrast sampling and bidirectional decoding require a reference policy. In our experiments, we used an [early checkpoint](https://drive.google.com/drive/u/0/folders/1FXHzPZPfTO7SCM-OTKUy3EEvKKTq3LC4) for the source of negative samples. One can direclty download the ["pretrained_weak_model"](https://drive.google.com/drive/u/0/folders/1kBPDBcPU3gLYCZNxkoRpXQ_e3tqi9lPR) and use it as the reference policy. Download and save the weak pre-trained checkpoint as ```lerobot/ckpt/pretrained_weak_model```. 

To use random sampling, run the following command:
```
bash commands/eval_random_sampler.sh
```
This script requires: 
(1) a pre-trained policy. We use the [LeRobot pre-trained checkpoint](https://huggingface.co/lerobot/vqbet_pusht). 
(2) specify the action horizon to be used. For example, for closed-loop sampling, we use ```--ah_test=1```. For open-loop sampling, we use ```aH_test=5```. 
(3) the temperature to be used in the softmax layer. For example, ```--temperature=0.5``` 
(4) the noise level in the environment. For example, for a deterministic environment we use ```--noise_level=0.0```.

To use backward coherence sampling, run the following command:
```
bash commands/eval_coherence_sampler.sh
```
This requires the same arguments as random sampling. 

To use forward contrast sampling, run the following command:
```
bash commands/eval_contrastive_sampler.sh
```
This requires the following additional arguments:
(1) a weak policy for contrast. We use the [early checkpoint](https://drive.google.com/drive/u/0/folders/1FXHzPZPfTO7SCM-OTKUy3EEvKKTq3LC4). 


To use bidirectional decoding, run the following command:
```
bash commands/eval_bidirectional_sampler.sh
```
This requires the same arguments as forward contrast. 

## Expected results 

We provide the expected success rate for the comparison of vanilla sampling and BID (both open and closed loop):

|              | Vanilla Open Loop | BID Open Loop | Vanilla Closed Loop | BID Closed Loop |
|--------------|-------------------|---------------|---------------------|-----------------|
| Noise = 0.0  | 61.0              | **65.2**      | 52.0                | **56.6**        |
| Noise = 1.0  | 39.0              | 39.8          | 50.4                | **54.8**        |
| Noise = 1.5  | 19.4              | 21.4          | 44.2                | **54.4**        |

For vanilla sampling, we use the default LeRobot ```temperature=0.1```. For BID, we used ```temperature=0.5``` to ensure sampling diversity. 

Furthermore, to support our theoretical analysis, we evaluate the performance of various action horizons under vanilla sampling in clean and noisy environments. The expected success rates are:

| Horizon          |   1     |   3  |   5  |
|------------------|---------|------|------|
| Noise = 0.0      |  52.0   | 57.6 | 61.0 |
| Noise = 1.5      |  44.2   | 43.4 | 19.4 |

We also evaluate the performance of each of the forward contrast and backward coherence components of BID in the highly noisy setting with ```Noise = 1.5```. The expected success rates are:

| Method           |   Backward Coherence     |   Forward Contrast  |   Bidirectional Decoding  |
|------------------|--------------------------|---------------------|---------------------------|
| Noise = 1.5      |        46.8              |          47.4       |         54.4              |
