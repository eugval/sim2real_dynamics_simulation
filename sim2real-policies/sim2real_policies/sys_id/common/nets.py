"""
commonly used networks for system identification functions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  # add parent path



class BaseNetwork(nn.Module):
    """ Base network class """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(BaseNetwork, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim

    def forward(self,):
        pass
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval().cuda()

class ProjectionNetwork(BaseNetwork):
    def __init__(self, input_dim, output_dim, hidden_dim=64, activation = F.relu, output_activation = F.tanh):
        """ 
        Compress the dynamics parameters to low-dimensional
        https://arxiv.org/abs/1903.01390 
        """
        super().__init__(input_dim, output_dim, hidden_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.linear4 = nn.Linear(int(hidden_dim/4), output_dim)
        self.activation = activation
        self.output_activation = output_activation

    def forward(self, input):
        if isinstance(input, np.ndarray) and len(input.shape) < 2:
            input = torch.FloatTensor(np.expand_dims(input, 0)).cuda()
        x = self.activation(self.linear1(input))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.output_activation(self.linear4(x))
        return x

    def get_context(self, input):
        """
        for inference only, no gradient descent
        """
        context = self.forward(input)
        return context.squeeze(0).detach().cpu().numpy()


class PredictionNetwork(BaseNetwork):
    """ 
    prediction model to predict next state given current state and action 
    (plus the embedding of dynamics if conditioned on EPI) 
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__(input_dim, output_dim, hidden_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, activation=F.relu, output_activation=F.tanh):
        if isinstance(input, np.ndarray) and len(input.shape) < 2:
            input = torch.FloatTensor(np.expand_dims(input, 0)).cuda()
        x = activation(self.linear1(input))
        x = activation(self.linear2(x))
        x = activation(self.linear3(x))
        x = output_activation(self.linear4(x))
        return x.squeeze(0)

class EmbedNetwork(BaseNetwork):
    """
    Embedding network for compress trajectories from EPI policy to low-dimensional embeddings
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__(input_dim, output_dim, hidden_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, activation=F.relu, output_activation=F.tanh):
        if isinstance(input, np.ndarray) and len(input.shape) < 2:
            input = torch.FloatTensor(np.expand_dims(input, 0)).cuda()
        x = activation(self.linear1(input))
        x = activation(self.linear2(x))
        x = output_activation(self.linear3(x))
        return x.squeeze(0)