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

def bidirectional_sampler(strong, weak, prior, obs_dict, ah_count, ah_test, temperature=1.0, num_sample=20, beta=0.90, factor=2):
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
    
    # generating new action chunk - do bidirectional behavior sampling
    
    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)

    action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
    action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)

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


def bidirectional_sampler_latent(strong_agent, strong, weak, prior, prior_action, obs_dict, ah_count, ah_test, temperature=1.0, num_sample=40, beta=1.0, factor=2):
    """
    Sample an action that preserves coherence with a prior and contrast outputs from strong and weak policies.
    Args:
        strong agent: a strong policy to predict near-optimal sequences of actions. This policy will 
        predict action chunk using the latent embeddings (i.e. sampled centers) found using BID 
        strong: copy of strong agent which will be used to find the optimal latent embeddings (i.e. sampled centers)
        weak: a weak policy which will be used to find the optimal latent embeddings (i.e. sampled centers)
        prior: the prior latent made in the previous time step. IMPORTANT: this prior is prior latent embeddings, not prior action chunk. 
        prior_action: the prior action chunk predicted
        obs_dict: dictionary containing observations at the current time step
        ah_count: the number of steps into the action chunk. ah_count takes on values between 0,...,AH - 1. Only when ah_count = 0, we will compute new action chunk using BID.
        ah_test: action horizon. This takes on values between 1,...,AH.
        temperature: temperature for sampling from VQ-BeT
        num_sample: number of samples to generate
        beta: beta for weighted distance calculation
        factor: Factor to determine the number of top samples to consider

    Returns:
        dict: A dictionary of actions sampled using BID.
    """ 

    B, OD = obs_dict['observation.state'].shape

    # pre-process
    # we need to repeat each observation for num_sample times to predict num_sample action chunks for BID
    obs_dict_batch = {
        'observation.state': obs_dict['observation.state'].unsqueeze(1).repeat(1, num_sample, 1).view(B * num_sample, OD),
        'observation.image': obs_dict['observation.image'].unsqueeze(1).repeat(1, num_sample, 1, 1, 1).view(B * num_sample, 3, 96, 96)
    }


    # predict
    # predict num_sample action chunks using strong and weak policies
    # we predict action chunks even if we are inside the action chunk to update the history of observations for the strong and weak policies
    action_strong, action_strong_chunk, latent_strong = strong.select_action(obs_dict_batch, ah_test, temperature)
    action_weak, action_weak_chunk, latent_weak = weak.select_action(obs_dict_batch, ah_test, temperature)    
    action_strong = action_strong.unsqueeze(1)
    action_weak = action_weak.unsqueeze(1)

    # prepare the dictionaries for strong and weak policies
    action_strong_batch = dict()
    action_strong_batch['action'] = action_strong
    action_strong_batch['action_pred'] = action_strong_chunk
    action_strong_batch['latent'] = latent_strong
    action_weak_batch = dict()
    action_weak_batch['action'] = action_weak
    action_weak_batch['action_pred'] = action_weak_chunk
    action_weak_batch['latent'] = latent_weak

    # check if we are inside an action chunk. If so, we do not need to take the new action predicted and instead execute the prior.
    if ah_count != 0:
        # we are inside an action chunk. No need to compute new action - take action that we already have to take. 
        # take action using prior
        action_strong, action_strong_chunk, latent_strong = strong_agent.select_action(obs_dict_batch_original, ah_test, temperature)
        
        action_dict = dict()
        action_dict['action'] = prior[:, :1, :]
        action_dict['action_pred'] = prior
        action_dict['latent'] = prior
        return action_dict
    
    # we are not inside an action chunk. We need to compute new action chunk using BID.
    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]
    latent_AH, latent_PH, latent_AD = latent_strong.shape[1], latent_strong.shape[1], latent_strong.shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)
    action_strong_batch['latent'] = action_strong_batch['latent'].reshape(B, num_sample, latent_AH, latent_AD)
    action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
    action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)
    action_weak_batch['latent'] = action_weak_batch['latent'].reshape(B, num_sample, latent_AH, latent_AD)

    

    # Backward coherence in latent space:
    if prior is not None:

        # distance measure
        CH = prior.shape[1]
        num_sample = num_sample // factor
        weights = torch.tensor([beta**i for i in range(CH)]).to(prior.device)
        weights = weights / weights.sum()

        # sample selection
        # dist_strong = euclidean_distance(action_strong_batch['latent'][:, :, :CH].float(), prior.unsqueeze(1).float(), reduction='none')
        dist_strong = manhattan_distance(action_strong_batch['latent'][:, :, :CH].float(), prior.unsqueeze(1).float(), reduction='none')
        # dist_strong = cosine_distance(action_strong_batch['latent'][:, :, :CH].float(), prior.unsqueeze(1).float(), reduction='none')
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
        # dist_weak = euclidean_distance(action_weak_batch['latent'][:, :, :CH].float(), prior.unsqueeze(1).float(), reduction='none')
        dist_weak = manhattan_distance(action_weak_batch['latent'][:, :, :CH].float(), prior.unsqueeze(1).float(), reduction='none')
        # dist_weak = cosine_distance(action_weak_batch['latent'][:, :, :CH].float(), prior.unsqueeze(1).float(), reduction='none')
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

    
    # forward contrast in latent space:
    src_latent_expand = action_strong_batch['latent'].unsqueeze(1).float()
    tar_latent_expand =  action_strong_batch['latent'].unsqueeze(2).float()
    # dist_latent_pos = euclidean_distance(src_latent_expand, tar_latent_expand).view(B, num_sample, num_sample)
    dist_latent_pos = manhattan_distance(src_latent_expand, tar_latent_expand).view(B, num_sample, num_sample)
    # dist_latent_pos = cosine_distance(src_latent_expand, tar_latent_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_latent_pos, k=topk, largest=False, dim=-1)
    dist_avg_latent_pos = values[:, :, 1:].mean(dim=-1)      # skip the self-distance first element

    src_latent_expand = action_strong_batch['latent'].unsqueeze(1).float()
    tar_latent_expand = action_weak_batch['latent'].unsqueeze(2).float()
    # dist_latent_neg = euclidean_distance(src_latent_expand, tar_latent_expand).view(B, num_sample, num_sample)
    dist_latent_neg = manhattan_distance(src_latent_expand, tar_latent_expand).view(B, num_sample, num_sample)
    # dist_latent_neg = cosine_distance(src_latent_expand, tar_latent_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_latent_neg, k=topk, largest=False, dim=-1)
    dist_avg_latent_neg = values[:, :, 0:].mean(dim=-1)

    # sample selection
    dist_avg_latent = dist_avg_latent_pos - dist_avg_latent_neg
    _, index_latent = dist_avg_latent.min(dim=-1)

    # optimal latent 
    range_tensor = torch.arange(B, device=index_latent.device)
    sampled_centers = action_strong_batch['latent'][range_tensor, index_latent]
    sampled_centers = sampled_centers.reshape(-1, AD)


    obs_dict_batch_original = {
        'observation.state': obs_dict['observation.state'].unsqueeze(1).repeat(1, 1, 1).view(B * 1, OD),
        'observation.image': obs_dict['observation.image'].unsqueeze(1).repeat(1, 1, 1, 1, 1).view(B * 1, 3, 96, 96)
    }
    action_strong, action_strong_chunk, latent_strong = strong_agent.select_action(obs_dict_batch_original, ah_test, temperature, sampled_centers)
    action_dict = dict()
    action_dict['action'] = action_strong
    action_dict['action_pred'] = action_strong_chunk
    action_dict['latent'] = latent_strong

    return action_dict

def bidirectional_plus_ema_sampler(strong, weak, prior, obs_dict, ah_count, ah_test, temperature=0.5, num_sample=20, beta_bid=0.99, factor=2, beta_ema=0.4):
    """
    Sample an action that preserves coherence with a prior and contrast outputs from strong and weak policies.
    Then, we take the moving average of the action with the prior.
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
    
    # generating new action chunk - do bidirectional behavior sampling
    
    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)

    action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
    action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)

    # backward
    if prior is not None:
        # distance measure
        CH = prior.shape[1]
        num_sample = num_sample // factor
        weights = torch.tensor([beta_bid**i for i in range(CH)]).to(prior.device)
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

    if prior is not None:
        # Extract padding values from action_dict_batch['action_pred']
        padding_values = action_dict['action_pred'][:,prior.shape[1]:, :]
            
        # Ensure prior and padding_values are aligned in dimensions
        prior_padded = torch.cat([prior, padding_values], dim=1)
        
        action_dict['action_pred'] = prior_padded * beta_ema + action_dict['action_pred'] * (1 - beta_ema)
        action_dict['action'] = action_dict['action_pred'][:,0,:]

    return action_dict
