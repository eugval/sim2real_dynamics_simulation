import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, action_range):
        super(PolicyNetworkBase, self).__init__()
        if isinstance(state_space, int): # pass in state_dim rather than state_space
            self._state_dim = state_space
        else:
            self._state_space = state_space
            self._state_shape = state_space.shape
            if len(self._state_shape) == 1:
                self._state_dim = self._state_shape[0]
            else:  # high-dim state
                pass  
        
        if isinstance(action_space, int): # pass in action_dim rather than action_space
            self._action_dim = action_space
        else:
            self._action_space = action_space
            self._action_shape = action_space.shape
            self._action_dim = self._action_shape[0]
        self.action_range = action_range

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self,):
        a=torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range*a.numpy()

class PPO_PolicyNetwork(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, self._action_dim)

        # self.log_std_linear = nn.Linear(hidden_dim, self._action_dim)
        # not dependent on latent features, reference:https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py
        self.log_std = AddBias(torch.zeros(self._action_dim))  

        self.action_range = action_range

        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = self.action_range * F.tanh(self.mean_linear(x))
        # implementation 1
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    
        # implementation 2
        zeros = torch.zeros(mean.size())
        if state.is_cuda:
            zeros = zeros.cuda()
        log_std = self.log_std(zeros)

        std = log_std.exp()
        return mean, std
        
    def get_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        mean, std = self.forward(state)
        if deterministic:
            action = mean
        else:
            pi = torch.distributions.Normal(mean, std)
            action = pi.sample()
        action = torch.clamp(action, -self.action_range, self.action_range)
        return action.detach().cpu().numpy()[0]

    def sample_action(self,):
        a=torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return a.numpy()

class DPG_PolicyNetwork(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, action_range=1., init_w=3e-3):
        super().__init__(state_space, action_space, action_range)
        
        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim) 

        self.output_linear = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action
        # weights initialization
        self.output_linear.weight.data.uniform_(-init_w, init_w)
        self.output_linear.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, hidden_activation=F.relu, output_activation=F.tanh):
        x = hidden_activation(self.linear1(state)) 
        x = hidden_activation(self.linear2(x))
        x = hidden_activation(self.linear3(x))
        x = hidden_activation(self.linear4(x))
        output  = output_activation(self.output_linear(x))
        return output

    def evaluate(self, state, noise_scale=0.5):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        action = self.forward(state)
        ''' add noise '''
        normal = Normal(0, 1)
        eval_noise_clip = 2*noise_scale
        noise = normal.sample(action.shape) * noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = self.action_range*action + noise.cuda()
        return action


    def get_action(self, state, noise_scale=0.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).cuda() # state dim: (N, dim of state)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0] 
        ''' add noise '''
        normal = Normal(0, 1)
        noise = noise_scale * normal.sample(action.shape)
        action=self.action_range*action + noise.numpy()

        return action


class DPG_PolicyNetworkLSTM(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, action_range=1., init_w=3e-3):
        super().__init__(state_space, action_space, action_range)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, self.hidden_dim) # output dim = dim of action

        self.output_linear = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action
        # weights initialization
        self.output_linear.weight.data.uniform_(-init_w, init_w)
        self.output_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        activation=F.relu
        # branch 1
        fc_branch = activation(self.linear1(state)) 
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = activation(self.linear2(lstm_branch))   # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        lstm_branch,  lstm_hidden = self.lstm1(lstm_branch, hidden_in)    # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1)   
        x = activation(self.linear3(merged_branch))
        x = activation(self.linear4(x))
        output  = F.tanh(self.output_linear(x))
        # x = F.tanh(self.linear4(x)).clone()
        output = output.permute(1,0,2)  # permute back

        return output, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

    def evaluate(self, state, last_action, hidden_in, noise_scale=0.5):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        action, hidden_out = self.forward(state, last_action, hidden_in)
        ''' add noise '''
        normal = Normal(0, 1)
        eval_noise_clip = 2*noise_scale
        noise = normal.sample(action.shape) * noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = self.action_range*action + noise.cuda()
        return action, hidden_out

    def get_action(self, state, last_action, hidden_in, noise_scale=0.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda() # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        action, hidden_out = self.forward(state, last_action, hidden_in)
        action = action.detach().cpu().numpy()[0][0]
        ''' add noise '''
        normal = Normal(0, 1)
        noise = noise_scale * normal.sample(action.shape)
        action=self.action_range*action + noise.numpy()
        return action , hidden_out

        # normal = Normal(0, 1)
        # noise = noise_scale * normal.sample(action.shape).cuda()
        # action=self.action_range*action + noise
        # return action.detach().cpu().numpy()[0][0], hidden_out



class SAC_PolicyNetwork(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample()
        action_0 = torch.tanh(mean + std * z.cuda())  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.cuda()) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample().cuda()
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action



class SAC_PolicyNetworkLSTM(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = F.relu(self.linear1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, lstm_hidden
    
    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample()
        action_0 = torch.tanh(mean + std * z.cuda())  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.cuda()) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()  # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample().cuda()
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy() if deterministic else \
        action.detach().cpu().numpy()
        return action[0][0], hidden_out


class RandomPolicy():
    def __init__(self, action_dim, action_range=1.):
        self.action_dim = action_dim
        self.action_range = action_range
    
    def get_action(self, state):
        random_action = np.random.uniform(-self.action_range, self.action_range, self.action_dim)
        return random_action

    def save_model(self, path):
        pass
    
    def load_model(self, path):
        pass
    
    def share_memory(self):
        pass
    
    def choose_action(self, state):
        return self.get_action(state)

    def to_cuda(self):
        pass

