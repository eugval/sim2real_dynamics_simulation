import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  # add parent path

from utils.policy_networks import SAC_PolicyNetwork, DPG_PolicyNetwork, PPO_PolicyNetwork
from sys_id.common.nets import ProjectionNetwork
import numpy as np
from gym import spaces
import torch


def load(path='../../assets/rl/sac/model/sac_v2_multiprocess_multi', alg='SAC', state_dim=9, action_dim=2):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
    action_space = spaces.Box(low=-1., high=1., shape=(action_dim,))
    if alg=='SAC':
        policy_net = SAC_PolicyNetwork(state_space, action_space, hidden_size=512, action_range=1.)
    elif alg=='TD3':
        policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim=512, action_range=1.)
    elif alg=='PPO':
        policy_net = PPO_PolicyNetwork(state_space, action_space, hidden_dim=512, action_range=1.)
    else:
        raise NotImplementedError
    policy_net.load_state_dict(torch.load(path + '_policy', map_location='cuda:0'))
    policy_net.eval().cuda()

    return policy_net  # call .get_action(state) to return action


def load_model(path, input_dim, output_dim):
    proj_net = ProjectionNetwork(input_dim, output_dim)

    proj_net.load_state_dict(torch.load(path, map_location='cuda:0'))
    proj_net.eval().cuda()

    return proj_net  # call .get_action(state) to return action


