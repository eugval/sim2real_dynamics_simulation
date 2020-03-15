import os
from sim2real_policies.utils.policy_networks import *
from sim2real_policies.sys_id.universal_policy_online_system_identification.osi import OSINetwork
from sim2real_policies.sys_id.common.nets import PredictionNetwork,  EmbedNetwork
import numpy as np
from gym import spaces
import torch


def load(path=None, alg='TD3', state_dim=9, action_dim=2):
    if path is None:
        raise NotImplementedError
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
    action_space = spaces.Box(low=-1., high=1., shape=(action_dim,))
    if alg=='SAC' or alg == 'sac':
        policy_net = SAC_PolicyNetwork(state_space, action_space, hidden_size=512, action_range=1.)
    elif alg=='TD3' or alg == 'td3':
        policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim=512, action_range=1.)
    elif alg=='PPO' or alg == 'ppo':
        policy_net = PPO_PolicyNetwork(state_space, action_space, hidden_dim=512, action_range=1.)
    elif alg=='TD3LSTM' or alg == 'lstm_td3':
        policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim=512)
    elif alg=='TD3OSI' or alg == 'uposi_td3':
        policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim=512, action_range=1.)
    elif alg=='TD3EPI' or alg == 'epi_td3':
        policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim=512, action_range=1.)
    else:
        raise NotImplementedError
    if alg=='PPO' or alg == 'ppo':
        policy_net.load_state_dict(torch.load(path + '_actor', map_location='cuda:0'))
    else:
        policy_net.load_state_dict(torch.load(path + '_policy', map_location='cuda:0'))
    policy_net.eval().cuda()

    return policy_net  # call .get_action(state) to return action



def load_model(model_name, path, input_dim, output_dim, hidden_dim=512):
    if model_name == 'osi':
        model = OSINetwork(input_dim = input_dim, output_dim = output_dim)
    elif model_name == 'embedding':
        model = EmbedNetwork(input_dim = input_dim, output_dim = output_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(path+'_'+model_name))
    model.eval()

    return model  # call .get_action(state) to return action

