'''
Twin Delayed DDPG (TD3) with LSTM branch for encoding environment dynamics from history state-action pairs.
Not just LSTM TD3, but follows the paper of domain randomisation with LSTM policy:
https://arxiv.org/pdf/1710.06537.pdf
'''
import math
import random

import gym
import numpy as np

import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work

import argparse
import torch.multiprocessing as mp

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from sim2real_policies.utils.envs import make_env
from sim2real_policies.utils.choose_env import choose_env
from sim2real_policies.utils.optimizers import SharedAdam
from sim2real_policies.utils.buffers import *
from sim2real_policies.utils.value_networks import *
from sim2real_policies.utils.policy_networks import *
from sim2real_policies.utils.evaluate import evaluate_lstm
from sim2real_policies.utils.load_params import load_params
from sim2real_policies.td3.td3_multiprocess import TD3_Trainer

from mujoco_py import MujocoException


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class TD3_Trainer_LSTM(TD3_Trainer):
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        action_range, policy_target_update_interval=1):
        super().__init__(replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        action_range, policy_target_update_interval)

        self.q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim)
        self.q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim)
        self.target_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim)
        self.target_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim)
        self.policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim)
        self.target_policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
    
        # needs to define optimizers here for new networks to replace the optimizers in parent class
        self.q_optimizer1 = SharedAdam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = SharedAdam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = SharedAdam(self.policy_net.parameters(), lr=policy_lr)

    def update(self, batch_size, eval_noise_scale, deterministic=True, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        ''' overlap the td3_training update function in a lstm manner '''
        # print('sample:', state, action,  reward, done)

        # uncomment to put (hidden, cell) on correct gpu if using multiple gpus
        # (hi, ci) = hidden_in
        # hi, ci = hi.cuda(), ci.cuda()
        # hidden_in = (hi, ci)
        # (ho, co) = hidden_out
        # ho, co = ho.cuda(), co.cuda()
        # hidden_out = (ho, co)

        state      = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action     = torch.FloatTensor(action).cuda()
        last_action     = torch.FloatTensor(last_action).cuda()
        reward     = torch.FloatTensor(reward).unsqueeze(-1).cuda()
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).cuda()
        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in)
        new_action,  _= self.policy_net.evaluate(state, last_action, hidden_in, eval_noise_scale)  # no noise, deterministic policy gradients
        new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out, eval_noise_scale) # clipped normal noise

        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        predicted_target_q1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_out)
        predicted_target_q2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)

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
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1

        return predicted_q_value1.mean()  # for debug

def worker(id, td3_trainer, env_name, environment_params, environment_wrappers, environment_wrapper_arguments, rewards_queue, eval_rewards_queue, success_queue,\
            eval_success_queue, eval_interval, replay_buffer, max_episodes, max_steps, batch_size, explore_steps, noise_decay, update_itr, explore_noise_scale, \
            eval_noise_scale, reward_scale, DETERMINISTIC, hidden_dim, model_path, seed=1):
    '''
    the function for sampling with multi-processing
    '''
    with torch.cuda.device(0):
        td3_trainer.to_cuda()
        print(td3_trainer, replay_buffer)
        env= make_env('robosuite.'+env_name, seed, id, environment_params, environment_wrappers,environment_wrapper_arguments)()
        action_dim = env.action_space.shape[0]
        frame_idx=0
        rewards=[]
        current_explore_noise_scale = explore_noise_scale
        # training loop
        for eps in range(max_episodes):
            last_action = env.action_space.sample()
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []
            # hidden_out = (torch.zeros([1, 1, hidden_dim]).type(torch.FloatTensor).cuda(), \
            #     torch.zeros([1, 1, hidden_dim]).type(torch.FloatTensor).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())            
            state =  env.reset()
            current_explore_noise_scale = current_explore_noise_scale*noise_decay
            
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = td3_trainer.policy_net.get_action(state, last_action, hidden_in, noise_scale=current_explore_noise_scale)
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

                if step==0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out
                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(reward)
                episode_next_state.append(next_state)
                episode_done.append(done) 

                last_action = action
                state = next_state
                frame_idx += 1
                
                number_of_episodes = int(batch_size/env.horizon)  # as lstm uses episode-wise update
                if replay_buffer.get_length() > number_of_episodes:
                    for i in range(update_itr):
                        _=td3_trainer.update(number_of_episodes, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                
                if done:
                    # have to detach the tensor before push to the shared memory, otherwise error!
                    (h_i, c_i) = ini_hidden_in
                    (h_o, c_o) = ini_hidden_out
                    ini_hidden_in = (h_i.detach(), c_i.detach())
                    ini_hidden_out = (h_o.detach(), c_o.detach())
                    # only push to buffer with full length of samples in the episode
                    replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,\
                    episode_reward, episode_next_state, episode_done)
                    break

            print('Worker: ', id, '|Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward))
            rewards_queue.put(np.sum(episode_reward))
            success_queue.put(info['success'])
            if eps % eval_interval == 0 and eps > 0:
                # plot(rewards, id)
                td3_trainer.save_model(model_path)
                eval_r, eval_succ = evaluate_lstm(env, td3_trainer.policy_net, hidden_dim = hidden_dim)
                eval_rewards_queue.put(eval_r)
                eval_success_queue.put(eval_succ)

        td3_trainer.save_model(model_path)

if __name__ == '__main__':
    # hyper-parameters for RL training
    env_name = ['SawyerReach', 'SawyerPush', 'SawyerSlide'][1]
    max_episodes  = 9000
    max_steps   = 80 
    num_workers = 4  # or: mp.cpu_count()
    eval_interval = 100
    # load other default parameters
    [action_range, batch_size, explore_steps, update_itr, explore_noise_scale, eval_noise_scale, reward_scale, \
        hidden_dim, noise_decay, policy_target_update_interval, q_lr, policy_lr, replay_buffer_size, DETERMINISTIC] = \
            load_params('td3', ['action_range', 'batch_size', 'explore_steps', 'update_itr', 'explore_noise_scale',\
             'eval_noise_scale', 'reward_scale', 'hidden_dim', 'noise_decay', \
                 'policy_target_update_interval', 'q_lr', 'policy_lr','replay_buffer_size', 'deterministic'] )


    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBufferLSTM2)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(replay_buffer_size)  # share the replay buffer through manager

    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(env_name)
    prefix=env_name+str(len(environment_params["parameters_to_randomise"]))  # number of randomised parameters
    model_path = '../../../../data/lstm_td3/model/'+prefix+'_lstm_td3'

    action_space = env.action_space
    state_space = env.observation_space

    td3_trainer=TD3_Trainer_LSTM(replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        policy_target_update_interval=policy_target_update_interval, action_range=action_range )


    if args.train:
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
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            last_action = env.action_space.sample()

            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = td3_trainer.policy_net.get_action(state, last_action, hidden_in, noise_scale=0.0)
                next_state, reward, done, _ = env.step(action)
                env.render() 

                last_action = action
                episode_reward += reward
                state=next_state

                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
