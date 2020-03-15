'''
Twin Delayed DDPG (TD3), if no twin no delayed then it's DDPG.
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net, 1 target policy net
original paper: https://arxiv.org/pdf/1802.09477.pdf
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
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display

import argparse
import time
import queue

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from sim2real_policies.utils.envs import make_env
from sim2real_policies.utils.choose_env import choose_env
from sim2real_policies.utils.optimizers import SharedAdam, ShareParameters
from sim2real_policies.utils.buffers import ReplayBuffer
from sim2real_policies.utils.value_networks import *
from sim2real_policies.utils.policy_networks import DPG_PolicyNetwork
from sim2real_policies.utils.evaluate import evaluate
from sim2real_policies.utils.load_params import load_params

from mujoco_py import MujocoException

class TD3_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        action_range, policy_target_update_interval=1):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim

        self.q_net1 = QNetwork(state_space, action_space, hidden_dim)
        self.q_net2 = QNetwork(state_space, action_space, hidden_dim)
        self.target_q_net1 = QNetwork(state_space, action_space, hidden_dim)
        self.target_q_net2 = QNetwork(state_space, action_space, hidden_dim)
        self.policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim, action_range)
        self.target_policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
    
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = SharedAdam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = SharedAdam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = SharedAdam(self.policy_net.parameters(), lr=policy_lr)

    def to_cuda(self):
        self.q_net1 = self.q_net1.cuda()
        self.q_net2 = self.q_net2.cuda()
        self.target_q_net1 = self.target_q_net1.cuda()
        self.target_q_net2 = self.target_q_net2.cuda()
        self.policy_net = self.policy_net.cuda()
        self.target_policy_net = self.target_policy_net.cuda()
    
    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net
    
    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, 2done)

        state      = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action     = torch.FloatTensor(action).cuda()
        reward     = torch.FloatTensor(reward).unsqueeze(1).cuda()  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).cuda()

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action = self.policy_net.evaluate(state, noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action = self.target_policy_net.evaluate(next_state, noise_scale=eval_noise_scale) # clipped normal noise

        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

    # Training Q Function
        target_q_min = torch.min(self.target_q_net1(next_state, new_next_action),self.target_q_net2(next_state, new_next_action))

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()

        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1

        return predicted_q_value1.mean()

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1')
        torch.save(self.q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.q_net1.load_state_dict(torch.load(path+'_q1', map_location='cuda:0'))
        self.q_net2.load_state_dict(torch.load(path+'_q2', map_location='cuda:0'))
        self.policy_net.load_state_dict(torch.load(path+'_policy', map_location='cuda:0'))
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()

    def share_memory(self):
        self.q_net1.share_memory()
        self.q_net2.share_memory()
        self.target_q_net1.share_memory()
        self.target_q_net2.share_memory()
        self.policy_net.share_memory()
        self.target_policy_net.share_memory()
        ShareParameters(self.q_optimizer1)
        ShareParameters(self.q_optimizer2)
        ShareParameters(self.policy_optimizer)


def worker(id, td3_trainer, env_name, environment_params, environment_wrappers, environment_wrapper_arguments, rewards_queue, eval_rewards_queue, success_queue,\
            eval_success_queue, eval_interval, replay_buffer, max_episodes, max_steps, batch_size, explore_steps, noise_decay, update_itr, explore_noise_scale, \
            eval_noise_scale, reward_scale, DETERMINISTIC, hidden_dim, model_path, seed=1):
    '''
    the function for sampling with multi-processing
    '''
    with torch.cuda.device(id % torch.cuda.device_count()):
        td3_trainer.to_cuda()
        print(td3_trainer, replay_buffer)
        env= make_env('robosuite.'+env_name, seed, id, environment_params, environment_wrappers,environment_wrapper_arguments)()
        action_dim = env.action_space.shape[0]
        frame_idx=0
        rewards=[]
        current_explore_noise_scale = explore_noise_scale
        # training loop
        for eps in range(max_episodes):
            episode_reward = 0
            state =  env.reset()
            current_explore_noise_scale = current_explore_noise_scale*noise_decay
            
            for step in range(max_steps):
                if frame_idx > explore_steps:
                    action = td3_trainer.policy_net.get_action(state, noise_scale=current_explore_noise_scale)
                else:
                    action = td3_trainer.policy_net.sample_action()
        
                try:
                    next_state, reward, done, info = env.step(action)
                    if environment_params["has_renderer"] and environment_params["render_visual_mesh"]:
                        env.render()   
                except KeyboardInterrupt:
                    print('Finished')
                    td3_trainer.save_model(model_path)
                except MujocoException:
                    print('MujocoException')
                    break

                if info["unstable"]: # capture the case with cube flying away for pushing task
                    break

                replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                frame_idx += 1
                
                if replay_buffer.get_length() > batch_size:
                    for i in range(update_itr):
                        _=td3_trainer.update(batch_size, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                
                if done:
                    break
            print('Worker: ', id, '|Episode: ', eps, '| Episode Reward: ', episode_reward)
            rewards_queue.put(episode_reward)
            success_queue.put(info['success'])

            if eps % eval_interval == 0 and eps>0:
                # plot(rewards, id)
                td3_trainer.save_model(model_path)
                eval_r, eval_succ = evaluate(env, td3_trainer.policy_net)
                eval_rewards_queue.put(eval_r)
                eval_success_queue.put(eval_succ)

        td3_trainer.save_model(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)

    args = parser.parse_args()
    
    # hyper-parameters for RL training
    env_name = ['SawyerReach', 'SawyerPush', 'SawyerSlide'][1]  # 36000 episodes for SawyerPush, 28000 episodes for SawyerReach and SawyerSlide
    max_episodes  = 9000
    max_steps   = 80   # it doesn't matter as long as it's larger than env.horizon
    num_workers = 4 # or: mp.cpu_count()
    eval_interval = 100  # evaluate training processs every N episodes
    # load other default parameters
    [action_range, batch_size, explore_steps, update_itr, explore_noise_scale, eval_noise_scale, reward_scale, \
        hidden_dim, noise_decay, policy_target_update_interval, q_lr, policy_lr, replay_buffer_size, DETERMINISTIC] = \
            load_params('td3', ['action_range', 'batch_size', 'explore_steps', 'update_itr', 'explore_noise_scale',\
             'eval_noise_scale', 'reward_scale', 'hidden_dim', 'noise_decay', \
                 'policy_target_update_interval', 'q_lr', 'policy_lr','replay_buffer_size', 'deterministic'] )


    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(replay_buffer_size)  # share the replay buffer through manager

    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(env_name)
    prefix=env_name+str(len(environment_params["parameters_to_randomise"]))  # number of randomised parameters
    model_path = '../../../../data/td3/model/'+prefix+'_td3'

    action_space = env.action_space
    state_space = env.observation_space

    td3_trainer=TD3_Trainer(replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
         policy_target_update_interval=policy_target_update_interval, action_range=action_range )

    if args.train:   
        # curriculum learning, pre-train with non-randomised 
        # prefix=env_name+str(0)  # number of randomised parameters
        # ini_model_path = '../../../../data/td3/model/'+prefix+'_td3'
        # td3_trainer.load_model(ini_model_path)

        td3_trainer.load_model(model_path)
        td3_trainer.share_memory()

        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        eval_rewards_queue = mp.Queue()  # used for get offline evaluated rewards from all processes and plot the curve
        success_queue = mp.Queue()  # used for get success events from all processes
        eval_success_queue = mp.Queue()

        processes=[]
        rewards=[]
        success = []
        eval_rewards = []
        eval_success = []

        for i in range(num_workers):
            process = Process(target=worker, args=(i, td3_trainer, env_name, environment_params, environment_wrappers,environment_wrapper_arguments, \
            rewards_queue, eval_rewards_queue, success_queue, eval_success_queue, eval_interval, replay_buffer, max_episodes, max_steps, batch_size, explore_steps, noise_decay,\
            update_itr, explore_noise_scale, eval_noise_scale, reward_scale, DETERMINISTIC, hidden_dim, model_path))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep getting the episode reward from the queue
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
                np.save('eval_rewards', eval_rewards)
                np.save('eval_success', eval_success)


        [p.join() for p in processes]  # finished at the same time

        td3_trainer.save_model(model_path)
        
    if args.test:
        td3_trainer.load_model(model_path)
        td3_trainer.to_cuda()
        env.renderer_on()
        for eps in range(10):
            state =  env.reset()
            env.render()   
            episode_reward = 0

            for step in range(max_steps):
                action = td3_trainer.policy_net.get_action(state, noise_scale=0.0)
                next_state, reward, done, _ = env.step(action)
                env.render() 

                episode_reward += reward
                state=next_state

                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
