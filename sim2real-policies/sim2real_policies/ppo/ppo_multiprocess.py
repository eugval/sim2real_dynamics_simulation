'''
Multi-processing version of PPO continuous v1
'''


import math
import random

import gym
import numpy as np

import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import argparse
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import threading as td
from sim2real_policies.utils.value_networks import *
from sim2real_policies.utils.policy_networks import PPO_PolicyNetwork
from sim2real_policies.utils.choose_env import choose_env
from sim2real_policies.utils.envs import make_env
from sim2real_policies.utils.evaluate import evaluate
from sim2real_policies.utils.optimizers import SharedAdam, ShareParameters
from sim2real_policies.utils.initialize import AddBias
from sim2real_policies.utils.load_params import load_params

#####  hyper-parameters for RL training  ############
ENV_NAME = ['SawyerReach', 'SawyerPush', 'SawyerSlide'][0] # environment name
EP_MAX = 100000  # total number of episodes for training
EP_LEN = 200  # total number of steps for each episode
prefix=''
MODEL_PATH = '../../../../data/ppo/model/'+prefix+'ppo'
NUM_WORKERS=1  # or: mp.cpu_count()
EVAL_INTERVAL = 100

[ACTION_RANGE, BATCH, GAMMA, RANDOMSEED, A_UPDATE_STEPS, C_UPDATE_STEPS, EPS,\
    A_LR, C_LR, METHOD] = load_params('ppo', ['action_range', 'batch_size', 'gamma', 'random_seed', 'actor_update_steps', \
    'critic_update_steps', 'eps', 'actor_lr', 'critic_lr', 'method' ])

###############################  PPO  ####################################
        
class PPO(object):
    '''
    PPO class
    '''
    def __init__(self, state_space, action_space, hidden_dim=512):
        self.actor = PPO_PolicyNetwork(state_space, action_space, hidden_dim, action_range = ACTION_RANGE)
        self.critic = ValueNetwork(state_space, hidden_dim)
        self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=A_LR)
        self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=C_LR)
        print(self.actor, self.critic)
        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []

    def a_train(self, s, a, adv, oldpi):
        '''
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return:
        '''  
        mu, std = self.actor(s)
        pi = Normal(mu, std)

        # ratio = torch.exp(pi.log_prob(a) - oldpi.log_prob(a))  # sometimes give nan
        ratio = torch.exp(pi.log_prob(a)) / (torch.exp(oldpi.log_prob(a)) + EPS)
        surr = ratio * adv

        if METHOD['name'] == 'kl_pen':
            lam = METHOD['lam']
            kl = torch.distributions.kl.kl_divergence(oldpi, pi)
            kl_mean = kl.mean()
            aloss = -((surr - lam * kl).mean())
        else:  # clipping method, find this is better
            aloss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv))
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def c_train(self, cumulative_r, s):
        '''
        Update actor network
        :param cumulative_r: cumulative reward
        :param s: state
        :return: None
        '''
        v = self.critic(s)
        advantage = cumulative_r - v
        closs = (advantage**2).mean()
        self.critic_optimizer.zero_grad()
        closs.backward()
        self.critic_optimizer.step()


    def update(self, s=None, a=None, r=None):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        if s is None and a is None and r is None:  # EPIpolicy update
            s = torch.Tensor(self.state_buffer).cuda()
            a = torch.Tensor(self.action_buffer).cuda()
            r = torch.Tensor(self.cumulative_reward_buffer).cuda()
        else:  # task specific policy update
            s = torch.FloatTensor(s).cuda()  
            a = torch.FloatTensor(a).cuda()
            r = torch.FloatTensor(r).cuda()
    # def update(self):
    #     '''
    #     Update parameter with the constraint of KL divergent
    #     :return: None
    #     '''
    #     s = torch.Tensor(self.state_buffer).cuda()   
    #     a = torch.Tensor(self.action_buffer).cuda()   
    #     r = torch.Tensor(self.cumulative_reward_buffer).cuda()   
        with torch.no_grad():
            mean, std = self.actor(s)
            pi = torch.distributions.Normal(mean, std)
            adv = r - self.critic(s)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv, pi)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv, pi)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s) 

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()


    def choose_action(self, s, deterministic=False):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        a = self.actor.get_action(s, deterministic)
        return np.clip(a, -self.actor.action_range, self.actor.action_range)
    
    def get_v(self, s):
        '''
        Compute value
        :param s: state
        :return: value
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.FloatTensor(s).cuda()
        return self.critic(s).squeeze(0).detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path+'_actor')
        torch.save(self.critic.state_dict(), path+'_critic')

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path+'_actor'))
        self.critic.load_state_dict(torch.load(path+'_critic'))

        self.actor.eval()
        self.critic.eval()

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(torch.Tensor([next_state]).cuda()).cpu().detach().numpy()[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_   # no future reward if next state is terminal
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

    def to_cuda(self):
        self.actor = self.actor.cuda()
        self.critic = self.critic.cuda()

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        ShareParameters(self.actor_optimizer)
        ShareParameters(self.critic_optimizer)

def worker(id, ppo, environment_params, environment_wrappers,environment_wrapper_arguments, \
            rewards_queue, eval_rewards_queue, success_queue, eval_success_queue):
    with torch.cuda.device(id % torch.cuda.device_count()):
        ppo.to_cuda()
        env= make_env('robosuite.'+ENV_NAME, RANDOMSEED, id, environment_params, environment_wrappers, environment_wrapper_arguments)()
        
        all_ep_r = []
        for ep in range(EP_MAX):
            s = env.reset()
            ep_r = 0
            t0 = time.time()
            for t in range(EP_LEN):  # in one episode
                a = ppo.choose_action(s)
                s_, r, done, info = env.step(a)
                ppo.store_transition(s, a, r)
                s = s_
                ep_r += r

                if ENV_NAME == 'SawyerPush' and r < -2.5: # capture the case with cube flying away for pushing task
                    break

                if len(ppo.state_buffer) == BATCH:
                    ppo.finish_path(s_, done)
                    ppo.update()

                if done:
                    break
            ppo.finish_path(s_, done)
            all_ep_r.append(ep_r)

            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    ep, EP_MAX, ep_r,
                    time.time() - t0
                )
            )
            rewards_queue.put(ep_r)   
            success_queue.put(info['success'])  

            if ep % EVAL_INTERVAL == 0 and ep > 0:
                # plot(rewards, id)
                ppo.save_model(MODEL_PATH)
                eval_r, eval_succ = evaluate(env, ppo.actor)
                eval_rewards_queue.put(eval_r)
                eval_success_queue.put(eval_succ)   
        ppo.save_model(MODEL_PATH)

def main():
    # reproducible
    # env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)

    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(ENV_NAME)

    state_space = env.observation_space
    action_space = env.action_space

    ppo = PPO(state_space, action_space, hidden_dim=512)

    if args.train:
        ppo.share_memory()
        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        eval_rewards_queue = mp.Queue()  # used for get offline evaluated rewards from all processes and plot the curve
        success_queue = mp.Queue()  # used for get success events from all processes
        eval_success_queue = mp.Queue()
        processes=[]
        rewards=[]
        success = []
        eval_rewards = []
        eval_success = []

        for i in range(NUM_WORKERS):
            process = Process(target=worker, args=(i, ppo, environment_params, environment_wrappers,environment_wrapper_arguments,\
                rewards_queue, eval_rewards_queue, success_queue, eval_success_queue,))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep geting the episode reward from the queue
            # r = rewards_queue.get()
            # succ = success_queue.get()
            eval_r = eval_rewards_queue.get() # this queue has different sample frequence with above two queues, .get() at same time will break the while loop
            eval_succ = eval_success_queue.get() 

            # success.append(succ)
            # rewards.append(r)
            eval_rewards.append(eval_r)
            eval_success.append(eval_succ)

            if len(eval_rewards)%20==0 and len(eval_rewards)>0:
                # plot(rewards)
                # np.save(prefix+'td3_rewards', rewards)
                # np.save(prefix+'td3_success', success)
                np.save(prefix+'eval_rewards', eval_rewards)
                np.save(prefix+'eval_success', eval_success)

        [p.join() for p in processes]  # finished at the same time

        ppo.save_model(MODEL_PATH)


    if args.test:
        ppo.load_model(MODEL_PATH)
        ppo.to_cuda()
        while True:
            s = env.reset()
            for i in range(EP_LEN):
                env.render()
                a = ppo.choose_action(s, True)
                s, r, done, _ = env.step(a)
                if done:
                    break
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)

    args = parser.parse_args()
    main()
    