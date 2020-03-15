"""
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
import torch.optim as optim
import matplotlib.pyplot as plt


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
        for start, length in zip(splits, split_sizes))

def size_splits_in_two(tensor, first_size, dim=0):
    """Splits a tensor of size=first_size from the original tensor

        Arguments:
        tensor (Tensor): tensor to split.
        first_size (int): size of the first splitted tensor
        dim (int): dimension along which to split the tensor.
    """
    return size_splits(tensor, [first_size], dim)


def stack_data(traj, length):
    traj = np.array(traj)
    return traj[-length:, :].reshape(-1)


def plot(x, name='figure', path='./'):
    plt.figure(figsize=(20, 5))
    plt.plot(x)
    plt.savefig(path+name+'.png')
    # plt.show()
    plt.clf()


def plot_train_test(train_y, test_y, name='figure', path='./'):
    """ plot both the training and testing curves during training process """
    plt.figure(figsize=(20, 5))
    plt.plot(train_y, label='train')
    plt.plot(test_y, label='test')
    plt.legend()
    plt.grid()
    plt.savefig(path+name+'.png')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    env, _, _, _ = choose_env('SawyerReach')
    query_key_params(env)