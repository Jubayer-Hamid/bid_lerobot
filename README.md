# Bidirectional Action Decoding

This repository accompanies the paper **Bidirectional Action Decoding: Understanding and Improving Action Chunking for Generative Behavior Cloning**. 

We adapted the [LeRobot repository](https://github.com/huggingface/lerobot) to implement Bidirectional Action Decoding on the VQ-BeT policy. Our experiments run VQ-BeT on the PushT environment. 

Follow the setup instructions provided in the [LeRobot repository](https://github.com/huggingface/lerobot). 

## Evaluating pretrained policies
This repository allows one to evaluate a pretrained policy using (1) random sampling (2) backward decision coherence sampling (3) forward contrastive sampling and (4) bidirectional action decoding. 

To use random sampling, run the following command:
```
bash commands/eval_random_sampler.sh
```
This script requires: 
(1) a pre-trained policy. We use the [LeRobot pre-trained checkpoint](https://huggingface.co/lerobot/vqbet_pusht). 
(2) specify the action horizon to be used. For example, for closed loop sampling, we use ```--ah_test=1``` 
(3) the temperature to be used in the softmax layer. For example, ```--temperature=1.0``` 
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
(1) a weak policy for contrastive sampling. We use the following [early checkpoint](https://drive.google.com/drive/u/0/folders/1FXHzPZPfTO7SCM-OTKUy3EEvKKTq3LC4) in our experiments. 

To use bidirectional action decoding, run the following command:
```
bash commands/eval_bidirectional_sampler.sh
```
This requires the same arguments as contrastive sampling. 




## Citation

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
