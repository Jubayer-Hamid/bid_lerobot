import torch
from lerobot.common.samplers.metric import euclidean_distance
import ipdb

torch.set_printoptions(precision=1, sci_mode=False)

def coherence_sampler(policy, prior, observation, AH_count, AH_test, num_sample=20, beta=0.90):
    ''' 
    Generate multiple action samples from the policy based on the observation.
    If prior is provided, select the most coherent sample based on the prior.
    If prior is None, return any one of the generated samples.
    '''

    B = observation['observation.state'].shape[0] 
    
    # Initialize an empty dictionary to hold the processed observations
    observation_batch = dict()

    # Repeat and reshape 'observation.state'
    obs_state = observation['observation.state']
    obs_state_repeated = obs_state.unsqueeze(1).repeat(1, num_sample, 1).reshape(B * num_sample, -1)
    observation_batch['observation.state'] = obs_state_repeated

    # Repeat and reshape 'observation.image'
    obs_image = observation['observation.image']
    obs_image_repeated = obs_image.unsqueeze(1).repeat(1, num_sample, 1, 1, 1).reshape(B * num_sample, 3, 96, 96)
    observation_batch['observation.image'] = obs_image_repeated


    if prior is None:
        # generate the first action and action chunk of your trajectory. No coherence sampling needed here
        action, action_chunk = policy.select_action(observation_batch, AH_test)
        action = action.unsqueeze(1)
            
        action_dict_batch = dict()
        action_dict_batch['action'] = action
        action_dict_batch['action_pred'] = action_chunk

        # Reshape actions and action chunks back to the original batch size and number of samples
        AH, PH, AD = action_dict_batch['action'].shape[1], action_dict_batch['action_pred'].shape[1], action_dict_batch['action_pred'].shape[2]
        action_dict_batch['action'] = action_dict_batch['action'].reshape(B, num_sample, AH, AD)
        action_dict_batch['action_pred'] = action_dict_batch['action_pred'].reshape(B, num_sample, PH, AD)

        # Select the first action sample for each batch element since we are not doing coherence sampling here 
        action_dict = dict()
        for key in action_dict_batch.keys():
            action_dict[key] = action_dict_batch[key][:, 0]
        return action_dict

    if AH_count != 0:
        # so AH_count != 0 and we are inside an action chunk. No need to compute new action - take action that we already have to take. 
        action, action_chunk = policy.select_action(observation_batch, AH_test) # pass things through select_action just so that we store the observations and update the action queue 
        # select action that we had already planned on selecting. This will be the first element in the prior
        action_dict = dict()
        action_dict['action'] = prior[:, :1, :]
        action_dict['action_pred'] = prior
        return action_dict

    # Predict actions and action chunks
    action, action_chunk = policy.select_action(observation_batch, AH_test)
    action = action.unsqueeze(1)

    action_dict_batch = dict()
    action_dict_batch['action'] = action
    action_dict_batch['action_pred'] = action_chunk

    # Reshape actions and action chunks back to the original batch size and number of samples
    AH, PH, AD = action_dict_batch['action'].shape[1], action_dict_batch['action_pred'].shape[1], action_dict_batch['action_pred'].shape[2]
    action_dict_batch['action'] = action_dict_batch['action'].reshape(B, num_sample, AH, AD)
    action_dict_batch['action_pred'] = action_dict_batch['action_pred'].reshape(B, num_sample, PH, AD)

    # action_pred_avg = action_dict_batch['action_pred'].mean(dim=1, keepdim=True)
    # action_pred_diff = action_dict_batch['action_pred'] - action_pred_avg
    # action_pred_var = (action_pred_diff).abs().sum(dim=[1,2,3])
    # print('action_pred_var: ', action_pred_var.min().item(), action_pred_var.max().item(), action_pred_var.mean().item())

    # Calculate the distance if prior is provided
    CH = prior.shape[1]
    dist_raw = euclidean_distance(action_dict_batch['action_pred'][:, :, :CH], prior.unsqueeze(1), reduction='none')

    weights = torch.tensor([beta**i for i in range(CH)]).to(dist_raw.device)
    weights = weights / weights.sum()
    dist_weighted = dist_raw * weights.view(1, 1, CH)
    dist = dist_weighted.sum(dim=2)

    # Sample selection
    _, cross_index = dist.sort(descending=False)
    index = cross_index[:, 0]

    # Slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_dict_batch.keys():
        action_dict[key] = action_dict_batch[key][range_tensor, index]
    return action_dict

def random_sampler(policy, prior, observation, AH_count, AH_test, num_sample=20, beta=0.90):
    ''' 
    Given observation, take a random action. 
    '''

    B = observation['observation.state'].shape[0] 
    
    # Initialize an empty dictionary to hold the processed observations
    observation_batch = dict()

    # Repeat and reshape 'observation.state'
    obs_state = observation['observation.state']
    obs_state_repeated = obs_state.unsqueeze(1).repeat(1, num_sample, 1).reshape(B * num_sample, -1)
    observation_batch['observation.state'] = obs_state_repeated

    # Repeat and reshape 'observation.image'
    obs_image = observation['observation.image']
    obs_image_repeated = obs_image.unsqueeze(1).repeat(1, num_sample, 1, 1, 1).reshape(B * num_sample, 3, 96, 96)
    observation_batch['observation.image'] = obs_image_repeated


    if AH_count == 0: # if you want to do random sampling
        # generate the action and action chunk of your trajectory
        action, action_chunk = policy.select_action(observation_batch, AH_test)
        action = action.unsqueeze(1)
            
        action_dict_batch = dict()
        action_dict_batch['action'] = action
        action_dict_batch['action_pred'] = action_chunk

        # Reshape actions and action chunks back to the original batch size and number of samples
        AH, PH, AD = action_dict_batch['action'].shape[1], action_dict_batch['action_pred'].shape[1], action_dict_batch['action_pred'].shape[2]
        action_dict_batch['action'] = action_dict_batch['action'].reshape(B, num_sample, AH, AD)
        action_dict_batch['action_pred'] = action_dict_batch['action_pred'].reshape(B, num_sample, PH, AD)

        # Select the first action sample for each batch element since we are not doing coherence sampling here 
        action_dict = dict()
        for key in action_dict_batch.keys():
            action_dict[key] = action_dict_batch[key][:, 0]
        return action_dict

    if AH_count != 0:
        # so AH_count != 0 and we are inside an action chunk. No need to compute new action - take action that we already have to take. 
        action, action_chunk = policy.select_action(observation_batch, AH_test) # pass things through select_action just so that we store the observations and update the action queue 
        # select action that we had already planned on selecting. This will be the first element in the prior
        action_dict = dict()
        action_dict['action'] = prior[:, :1, :]
        action_dict['action_pred'] = prior
        return action_dict