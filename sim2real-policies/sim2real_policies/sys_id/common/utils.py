
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import os,sys,inspect
from gym import spaces
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  # add parent path
import torch.optim as optim
import matplotlib.pyplot as plt
from sim2real_policies.utils.choose_env import choose_env
from sim2real_policies.utils.policy_networks import DPG_PolicyNetwork, RandomPolicy

from mujoco_py import MujocoException

def query_key_params(env, normalize=True):
    """
    key parameters from heuristic: the proportional gains on 0, 1, 3, 5 joints
    """
    pass
    # joint_idx=[0]   #[0,1,3,5]
    # params_dict = env.get_dynamics_parameters()
    # params_ranges = env.get_parameter_sampling_ranges()
    # params_factors = env.get_factors_for_randomisation()
    # param_value = params_dict['kps']
    # if normalize:
    #     param_range = params_ranges['gains']
    #     param_factor = params_factors['kps']
    #     scale = param_range[1]-param_range[0]
    #     param_value = (param_value/(param_factor) - param_range[0])/scale
    # selected_param_value = param_value[joint_idx]
    
    # return np.array(selected_param_value)

def query_params(env, normalize=True, randomised_only=False, dynamics_only=False):
    """
    Query the dynamics parameters from the env, 
    normalize it if normalize is True,
    only return randomised parameters if randomised_only is True,
    only return dynamics parameters (no noise parameters) if dynamics_only is True.
    Return: array of parameters
    """
    params_dict = env.get_dynamics_parameters()
    randomised_params_keys = env.get_randomised_parameters()
    params_ranges = env.get_parameter_sampling_ranges()  # range
    params_factors = env.get_factors_for_randomisation()  # multiplied factor
    params_value_list = []
    for param_key, param_value in params_dict.items():
        if randomised_only and (param_key in randomised_params_keys) is False:
            continue
        if dynamics_only and ("time" in param_key or "noise" in param_key):
            # print('Non-dynamics parameters: ', param_key)
            continue
        else:
            if normalize:
                if param_key in params_factors.keys():
                    param_factor = params_factors[param_key]
                else:
                    raise NotImplementedError

                if param_key in params_ranges.keys():
                    param_range = params_ranges[param_key]
                else:
                    raise NotImplementedError
                
                scale = param_range[1]-param_range[0]
                param_factor = param_factor + 1e-15  # for factor=0.
                param_value = np.clip((param_value/(param_factor) - param_range[0])/(scale), 0., 1.)
                
            if isinstance(param_value, np.ndarray):
                params_value_list = list(np.concatenate((params_value_list, param_value)))
            else:   # scalar
                params_value_list.append(param_value)
    return np.array(params_value_list)

def _flatten_obs(obs_dict, verbose=False):  # gym observation wrapper
    """
    Filters keys of interest out and concatenate the information.

    Args:
        obs_dict: ordered dictionary of observations
    """
    keys = ["robot-state", "task-state", "target_pos",]
    ob_lst = []
    for key in obs_dict:
        if key in keys:  # hacked
            if verbose:
                print("adding key: {}".format(key))
            ob_lst.append(obs_dict[key])
    return np.concatenate(ob_lst)

def offline_history_collection(env_name, itr=30, policy=None, \
    vectorize=True, discrete=False, vine=False, vine_sample_size=500, egreedy=0):
    """ 
    Collect random simulation parameters and trajetories with given policy.
    ----------------------------------------------------------------
    params:
    env_name: name of env to collect data from
    itr: data episodes
    policy: policy used for collecting data
    vectorize: vectorized parameters into a list rather than a dictionary, used for system identification
    discrete: discrete randomisation range, as in EPI paper
    vine: Vine data collection, same state and same action at the initial of trajectory, as in EPI paper 
    vine_sample_size: number of state action samples in vine trajectory set
    egreedy: the factor for collecting data with epsilon-greedy policy
    """
    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(env_name)
    action_space = env.action_space
    state_space = env.observation_space
    if policy is None:  # load off-line policy if no policy
        policy=DPG_PolicyNetwork(state_space, action_space, 512).cuda()
        # load from somewhere
    
    history_sa=[]
    history_s_=[]
    params_list=[]
    if vine:
        vine_state_set = []  # underlying state of env, not the observation
        vine_action_set = [] # initial action after initial state
        vine_idx = 0
        # collect state action sets according to EPI's vine implementation
        while vine_idx<vine_sample_size:
            state =  env.reset()
            while vine_idx<vine_sample_size:
                if np.random.rand() < egreedy:
                    action = env.action_space.sample()
                else:
                    action = policy.get_action(state)
                vine_state_set.append(env.get_state())
                vine_action_set.append(action)  
                vine_idx += 1
                next_state, _, done, _ = env.step(action)
                state = next_state

                if done: break

    print('Start collecting transitions.')
    env.ignore_done = True
    for epi in range(itr):
        print('Episode: {}'.format(epi))
        state = env.reset()
        env.randomisation_off()
        # reset_params = env.get_dynamics_parameters()
        if discrete:
            env.randomisation_on()  # as sample_discretized_env_parameters() needs randomisation ranges
            sampled_env_params_dict = sample_discretized_env_parameters(env)
            env.randomisation_off()
            env.set_dynamics_parameters(sampled_env_params_dict)
        if vectorize:
            env.randomisation_on()
            params = query_params(env)
            env.randomisation_off()
        else:
            params = env.get_dynamics_parameters()
        params_list.append(params)

        if vine:
            epi_sa =[]
            epi_s_ =[]
            for underlying_state, action in zip(vine_state_set, vine_action_set):
                env.set_state(underlying_state)  # underlying state is different from obs of env
                state = _flatten_obs(env._get_observation())  # hacked
                try: 
                    next_state, _, done, _ = env.step(action)
                except MujocoException:
                    print('Data collection: MujocoException')
                    action = np.zeros_like(action)
                    next_state = state
                epi_sa.append(np.concatenate((state, action)))
                epi_s_.append(np.array(next_state))
                if done:   # keep using same env after done
                    env.reset()
            history_sa.append(np.array(epi_sa))
            history_s_.append(np.array(epi_s_))

        else:
            epi_traj = []
            for step in range(env.horizon):
                action = policy.get_action(state)
                epi_traj.append(np.concatenate((state, action)))
                try:
                    next_state, _, _, _ = env.step(action)
                except MujocoException:
                    print('MujocoException')
                    action = np.zeros(action)
                    next_state = state
                    continue   
                state = next_state
            history_sa.append(np.array(epi_traj))
        env.randomisation_on()
    if vine:
        history = [np.array(history_sa), np.array(history_s_)]
    else:
        history = np.array(history_sa)
    print("Finished collecting data.")
    return params_list, history


def OfflineSIupdate(input, label, epoch=500, lr=1e-1, save_path='./osi'):
    """ Update the system identification (SI) with offline dataset """
    criterion = nn.MSELoss()
    data_dim = len(input[0])
    label_dim = len(label[0])
    osi_model = OSINetork(input_dim = data_dim, output_dim = label_dim)
    optimizer = optim.Adam(osi_model.parameters(), lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # gamma: decay for each step
    input = torch.Tensor(input)
    label = torch.Tensor(label)

    for i in range(epoch):
        predict = osi_model(input)
        loss = criterion(predict, label)
        optimizer.zero_grad()
        loss.backward()
        print('Epoch: {} | Loss: {}'.format(epoch, loss))
        optimizer.step()
        scheduler.step()
    
    torch.save(osi_model.state_dict(), save_path)

def OfflineSI(env_name='SawyerReach'):    
    params, raw_history = offline_history_collection(env_name=env_name) 
    label, data = generate_data(params, raw_history)
    OfflineSIupdate(data, label)


def sample_discretized_env_parameters(env, splits=5, range_reduction_ratio=0.05):
    '''
    Uniformly sample parameters values from the valid randomisation range.
    return: parameter dictionary
    '''
    params_dict = env.get_dynamics_parameters()
    params_ranges = env.get_parameter_sampling_ranges()  # range
    params_factors = env.get_factors_for_randomisation() 
    randomized_params_list = env.get_randomised_parameters()
    for key in randomized_params_list:
        param_range = params_ranges[key]
        low = param_range[0]
        high = param_range[1]
        if isinstance(low, np.int) and isinstance(high, np.int):
            # is time-delayed parameter, already discrete in original env
            ranges = np.arrange(low, high+1)
            sampled_value = np.random.choice(ranges)
        else:
            value_range = high - low
            low = low + value_range*range_reduction_ratio
            high = high - value_range*range_reduction_ratio
            value_list = [low+((high-low)/splits)*i for i in range(splits)]
            sampled_value = params_factors[key]*np.random.choice(value_list)
        params_dict[key] = sampled_value

    return params_dict




if __name__ == '__main__':
    env, _, _, _ = choose_env('SawyerReach')
    params  =  query_params(env)
    print(params, params.shape)