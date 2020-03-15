"""
transition dataset collection with pretrained task-specific policy
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  # add parent path
from sim2real_policies.test.rl_utils import load, load_model
from utils.choose_env import choose_env
from sim2real_policies.sys_id.common.utils import offline_history_collection
from sim2real_policies.sys_id.common.operations import size_splits_in_two
import torch.optim as optim
from gym import spaces
import pickle
from sim2real_policies.utils.policy_networks import * 

from mujoco_py import MujocoException


def generate_transition_data(trajs, s_dim):
    # sa = trajs[:, :-1, :].reshape(-1, trajs.shape[-1])  # no last step in episode
    # s_ = trajs[:, 1:, :s_dim].reshape(-1, s_dim)  # no first step in episode
    sa = trajs[:, :-1, :]  # no last step in episode
    s_ = trajs[:, 1:, :s_dim]  # no first step in episode
    return sa, s_


def epi_data_collection(env_name, num_envs=100, train_data_ratio=0.8,\
    pretrained_policy=None, data_path='./data/', discrete=False, vine=False, egreedy=0.2):
    """
    Collect transition data for EPI algorithm with pre-trained policy 
    (trained with single env but interactcions with randomised envs).
    -------------------------------------------------------------
    Args:
    env_name: env to collect data from
    num_envs: number of randomized envs
    train_data_ratio: train/(train+test)
    pretrained_policy: policy for collecting data, using pre-trained task-specific policy in original EPI paper
    data_path: save path
    discrete: if true, discretize the randomization range with bins; else, randomize in continuous range
    vine: if true, collect in vine's manner as in original EPI paper, I personally think it's different from TRPO's vine
    egreedy: epsilon-greedy factor as in original EPI paper


    Return: 
    Collected transition data format: 
    {
    'x_train': (# episodes 1, # steps, state_dim + action_dim)
    'x_test': (# episodes 2, # steps, state_dim + action_dim)
    'y_train': (# episodes 1, # steps, state_dim)
    'y_test': (# episodes 2, # steps, state_dim)
    'param_train': (# episodes 1, # steps, param_dic)
    'param_test': (# episodes 2, # steps, param_dic)
    }
    """
    env, _, _, _ = choose_env(env_name)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    if pretrained_policy is None:  # load policy
        print('No policy provided!')
        pretrained_policy = load(path='', alg='PPO',state_dim=state_dim, action_dim=action_dim)
    # collect transition datasets
    params, trajectories = offline_history_collection(env_name, itr=num_envs, \
        policy=pretrained_policy, vectorize=False, discrete=discrete, vine=vine, vine_sample_size=500, egreedy=egreedy)
    print("Trajectories collected.")
    if vine:
        [state_action, next_state] = trajectories
    else:
        state_action, next_state = generate_transition_data(trajectories, state_dim)
    data={}
    assert len(state_action) == len(next_state) ==  len(params)
    split_pos = int(len(state_action)*train_data_ratio)  # split train and test
    data['x_train'] = state_action[:split_pos]
    data['x_test'] = state_action[split_pos:]
    data['y_train'] = next_state[:split_pos]
    data['y_test'] = next_state[split_pos:]
    data['param_train'] = params[:split_pos]
    data['param_test'] = params[split_pos:]
    if discrete:
        data_path += '_discrete'
    if vine:
        data_path += '_vine'
    data_path+='_data.pckl'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_path)
    pickle.dump(data,open(os.path.abspath(path),'wb' ))
    print('Processed data saved. Training data: {} episodes. Test data: {} episodes.'.format(len(data['x_train']),len(data['x_test'])))


if __name__ == '__main__':
    env_name = ['SawyerReach', 'SawyerPush', 'SawyerSlide'][1]
    env, _, _, _ = choose_env(env_name)
    action_space = env.action_space
    state_space = env.observation_space
    # policy = RandomPolicy(action_dim)
    # policy = PPO_PolicyNetwork(state_space, action_space, 512)
    policy  = DPG_PolicyNetwork(state_space, action_space, hidden_dim=512)
    prefix=env_name+str(0)  # task-specific policy over a fixed environment
    model_path = '../../../../../data/td3/model/'+prefix+'_td3'
    policy.load_state_dict(torch.load(model_path+'_policy', map_location='cuda:0'))  # load pre-trained model on fixed env
    policy.eval().cuda()
    epi_data_collection(env_name, num_envs=100, pretrained_policy=policy, data_path='./data/'+env_name, \
        discrete = True, vine=True, egreedy=0.2) # if discrete: discrete dynamics parameters for env
    


    
