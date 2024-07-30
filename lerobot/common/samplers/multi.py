import torch
from lerobot.common.samplers.metric import euclidean_distance, coverage_distance

import pdb
import ipdb
torch.set_printoptions(precision=2, sci_mode=False)

# prior, observation, count, ah_test

def contrastive_sampler(strong, weak, prior, obs_dict, ah_count, ah_test, temperature=1.0, num_sample=10, name='contrast', factor=2):
    """
    Sample an action by contrasting outputs from strong and weak policies.

    Args:
        strong: a strong policy to predict near-optimal sequences of actions
        weak: a weak policy to predict sub-optimal sequences of actions
        obs_dict: dictionary containing observations at the current time step
        num_sample (int, optional): number of samples to generate
        name (str, optional): type of samples ('contrast', 'positive', 'negative')
        factor (int, optional): Factor to determine the number of top samples to consider

    Returns:
        dict: A dictionary of actions sampled using the contrastive approach.
    """
    # pre-process
    B, OD = obs_dict['observation.state'].shape
    obs_dict_batch = {
        'observation.state': obs_dict['observation.state'].unsqueeze(1).repeat(1, num_sample, 1).view(B * num_sample, OD),
        'observation.image': obs_dict['observation.image'].unsqueeze(1).repeat(1, num_sample, 1, 1, 1).view(B * num_sample, 3, 96, 96)
    }

    # predict
    action_strong, action_strong_chunk = strong.select_action(obs_dict_batch, ah_test, temperature)
    action_weak, action_weak_chunk = weak.select_action(obs_dict_batch, ah_test, temperature)
    action_strong = action_strong.unsqueeze(1)
    action_weak = action_weak.unsqueeze(1)

    action_strong_batch = dict()
    action_strong_batch['action'] = action_strong
    action_strong_batch['action_pred'] = action_strong_chunk

    action_weak_batch = dict()
    action_weak_batch['action'] = action_weak
    action_weak_batch['action_pred'] = action_weak_chunk
    
    if ah_count != 0:
        # we are inside an action chunk. No need to compute new action - take action that we already have to take. 
        # take action using prior
        action_dict = dict()
        action_dict['action'] = prior[:, :1, :]
        action_dict['action_pred'] = prior
        return action_dict

    # generating new action chunk - do contrastive sampling
    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)

    action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
    action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)

    # positive samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand =  action_strong_batch['action_pred'].unsqueeze(2)
    dist_pos = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor + 1
    values, _ = torch.topk(dist_pos, k=topk, largest=False, dim=-1)
    dist_avg_pos = values[:, :, 1:].mean(dim=-1)      # skip the self-distance first element 

    if name == "negative": dist_avg_pos.zero_()

    # negative samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand = action_weak_batch['action_pred'].unsqueeze(2)
    dist_neg = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_neg, k=topk, largest=False, dim=-1)
    dist_avg_neg = values[:, :, 0:].mean(dim=-1)

    if name == "positive": dist_avg_neg.zero_()

    # sample selection
    dist_avg = dist_avg_pos - dist_avg_neg
    index = dist_avg.argmin(dim=-1)

    # slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_strong_batch.keys():
        action_dict[key] = action_strong_batch[key][range_tensor, index]
    return action_dict

def bidirectional_sampler(strong, weak, obs_dict, prior, temperature=1.0, num_sample=10, beta=0.99, factor=2):
    """
    Sample an action that preserves coherence with a prior and contrast outputs from strong and weak policies.
    Args:
        strong: a strong policy to predict near-optimal sequences of actions
        weak: a weak policy to predict sub-optimal sequences of actions
        prior: the prediction made in the previous time step
        obs_dict: dictionary containing observations at the current time step
        num_sample (int, optional): number of samples to generate
        name (str, optional): type of samples ('contrast', 'positive', 'negative')
        factor (int, optional): Factor to determine the number of top samples to consider

    Returns:
        dict: A dictionary of actions sampled using the contrastive approach.
    """    
    # pre-process
    B, OH, OD = obs_dict['obs'].shape
    obs_dict_batch = dict()
    for key in obs_dict.keys():
        if key == 'prior':
            continue        
        obs_dict_batch[key] = obs_dict[key].unsqueeze(1).repeat(1, num_sample, 1, 1).view(B * num_sample, OH, OD)

    # predict
    action_strong_batch = strong.predict_action(obs_dict_batch)
    action_weak_batch = weak.predict_action(obs_dict_batch)

    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_strong_batch:
        action_strong_batch['action_obs_pred'] = action_strong_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_strong_batch:
        action_strong_batch['obs_pred'] = action_strong_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
    action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_weak_batch:
        action_weak_batch['action_obs_pred'] = action_weak_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_weak_batch:
        action_weak_batch['obs_pred'] = action_weak_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    # backward
    if prior is not None:

        # distance measure
        CH = prior.shape[1]
        num_sample = num_sample // factor
        weights = torch.tensor([beta**i for i in range(CH)]).to(prior.device)
        weights = weights / weights.sum()

        # sample selection
        dist_strong = euclidean_distance(action_strong_batch['action_pred'][:, :, :CH], prior.unsqueeze(1), reduction='none')
        dist_weighted = dist_strong * weights.view(1, 1, CH)
        dist_strong_sum = dist_weighted.sum(dim=2)
        _, cross_index = dist_strong_sum.sort(descending=False)
        index = cross_index[:, 0:num_sample]

        # slicing
        action_dict = dict()
        range_tensor = torch.arange(B, device=index.device)
        for key in action_strong_batch.keys():
            action_dict[key] = action_strong_batch[key][range_tensor.unsqueeze(1), index]
        action_strong_batch = action_dict

        # sample selection
        dist_weak = euclidean_distance(action_weak_batch['action_pred'][:, :, :CH], prior.unsqueeze(1), reduction='none')
        dist_weighted = dist_weak * weights.view(1, 1, CH)
        dist_weak_sum = dist_weighted.sum(dim=2)
        _, cross_index = dist_weak_sum.sort(descending=False)
        index = cross_index[:, 0:num_sample]

        # slicing
        action_dict = dict()
        range_tensor = torch.arange(B, device=index.device)
        for key in action_weak_batch.keys():
            action_dict[key] = action_weak_batch[key][range_tensor.unsqueeze(1), index]
        action_weak_batch = action_dict

    # positive samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand =  action_strong_batch['action_pred'].unsqueeze(2)
    dist_pos = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_pos, k=topk, largest=False, dim=-1)
    dist_avg_pos = values[:, :, 1:].mean(dim=-1)      # skip the self-distance first element 

    # negative samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand = action_weak_batch['action_pred'].unsqueeze(2)
    dist_neg = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_neg, k=topk, largest=False, dim=-1)
    dist_avg_neg = values[:, :, 0:].mean(dim=-1)

    # sample selection
    dist_avg = dist_avg_pos - dist_avg_neg
    _, index = dist_avg.min(dim=-1)

    # slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_strong_batch.keys():
        action_dict[key] = action_strong_batch[key][range_tensor, index]

    return action_dict