'''
1. Universal Policy (UP): policy conditioned on the dynamics paramters
2. Online System Identification (OSI)
First UP, then OSI, same as original paper:
https://arxiv.org/abs/1702.02453 (Preparing for the Unknown: Learning a Universal Policy with Online System Identification)

Stages:
--up
--osi
--test


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
from gym import spaces

import argparse
import time
import queue

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from sim2real_policies.utils.envs import make_env
from sim2real_policies.utils.choose_env import choose_env
from sim2real_policies.utils.buffers import ReplayBuffer
from sim2real_policies.utils.evaluate import evaluate
from sim2real_policies.sys_id.common.utils import query_params
from sim2real_policies.sys_id.universal_policy_online_system_identification.osi import OSINetwork, OnlineSIupdate, generate_data, stack_data
from sim2real_policies.td3.td3_multiprocess import TD3_Trainer
from sim2real_policies.utils.load_params import load_params
from torch.utils.tensorboard import SummaryWriter

from mujoco_py import MujocoException

# writer = SummaryWriter()
# global configurations for parameters in up-osi
RANDOMISZED_ONLY = True  # only use randomised parameters
DYNAMICS_ONLY = True  # only use dynamics parameters
CAT_INTERNAL = True  # concatenate internal states with observation, and concatenate internal (joint) action with action; False only for sliding env

def worker(id, td3_trainer, osi_model, osi_l, osi_input_dim, osi_batch_size, osi_itr, osi_pretrain_eps, env_name, environment_params, environment_wrappers,\
             environment_wrapper_arguments, rewards_queue, eval_rewards_queue, success_queue,\
            eval_success_queue, osi_loss_queue, eval_interval, replay_buffer, max_episodes, max_steps, batch_size, explore_steps, noise_decay, update_itr, explore_noise_scale, \
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
        history_data = []
        params_list = []
        current_explore_noise_scale = explore_noise_scale
        # training loop
        for eps in range(max_episodes):
            episode_reward = 0
            epi_traj = []
            state =  env.reset()
            # attach the dynamics parameters as states
            params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY) # in reality, the true params cannot be queried
            params_state = np.concatenate((params, state))
            current_explore_noise_scale = current_explore_noise_scale*noise_decay
            
            for step in range(max_steps):
                # using internal state or not
                if CAT_INTERNAL:
                    internal_state = env.get_internal_state()
                    full_state = np.concatenate([state, internal_state])
                else:
                    full_state = state
                # exploration steps
                if frame_idx > explore_steps:
                    action = td3_trainer.policy_net.get_action(params_state, noise_scale=current_explore_noise_scale)
                else:
                    action = td3_trainer.policy_net.sample_action()
        
                try:
                    next_state, reward, done, info = env.step(action)
                    if environment_params["has_renderer"] and environment_params["render_visual_mesh"]:
                        env.render()   
                except KeyboardInterrupt:
                    print('Finished')
                    if osi_model is None:
                        td3_trainer.save_model(model_path)
                except MujocoException:
                    print('MujocoException')
                    break

                if info["unstable"]: # capture the case with cube flying away for pushing task
                    break

                if CAT_INTERNAL:
                    target_joint_action = info["joint_velocities"]
                    full_action = np.concatenate([action, target_joint_action])
                else:
                    full_action = action
                epi_traj.append(np.concatenate((full_state, full_action)))  # internal state for osi only, not for up


                if len(epi_traj)>=osi_l and osi_model is not None and eps > osi_pretrain_eps:
                    osi_input = stack_data(epi_traj, osi_l)  # stack (s,a) to have same length as in the model input
                    pre_params = osi_model(osi_input).detach().numpy()
                else:
                    pre_params = params

                if osi_model is not None and eps > osi_pretrain_eps:
                    if len(epi_traj)>=osi_l:
                        osi_input = stack_data(epi_traj, osi_l)  # stack (s,a) to have same length as in the model input
                        pre_params = osi_model(osi_input).detach().numpy()
                    else:
                        zero_osi_input = np.zeros(osi_input_dim)
                        pre_params = osi_model(zero_osi_input).detach().numpy()
                else:
                    pre_params = params

                next_params_state = np.concatenate((pre_params, next_state))
                replay_buffer.push(params_state, action, reward, next_params_state, done)
                
                state = next_state
                params_state = next_params_state
                episode_reward += reward
                frame_idx += 1
                
                if replay_buffer.get_length() > batch_size and osi_model is None:
                    for i in range(update_itr):
                        _=td3_trainer.update(batch_size, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                
                if done:
                    break

            if osi_model is not None: # train osi
                history_data.append(np.array(epi_traj))
                params_list.append(params)

                if eps % osi_batch_size == 0 and eps>0:
                    label, data = generate_data(params_list, history_data, length = osi_l)
                    osi_model, loss = OnlineSIupdate(osi_model, data, label, epoch=osi_itr)
                    osi_loss_queue.put(loss)
                    # clear the dataset; alternative: using a offline buffer
                    params_list = []
                    history_data = []
                    torch.save(osi_model.state_dict(), model_path+'_osi')
                    print('OSI Episode: {} | Epoch Loss: {}'.format(eps, loss))
            else: # train up
                print('Worker: ', id, '|Episode: ', eps, '| Episode Reward: ', episode_reward)
                rewards_queue.put(episode_reward)
                success_queue.put(info['success'])

                if eps % eval_interval == 0 and eps>0:
                    # plot(rewards, id)
                    td3_trainer.save_model(model_path)
                    eval_r, eval_succ = evaluate(env, td3_trainer.policy_net, up=True, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
                    eval_rewards_queue.put(eval_r)
                    eval_success_queue.put(eval_succ)

        if osi_model is not None: # train osi
            torch.save(osi_model.state_dict(), model_path+'_osi')
        else: # train up
            td3_trainer.save_model(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UP-OSI.')
    parser.add_argument('--up', dest='up', action='store_true', default=False)
    parser.add_argument('--osi', dest='osi', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)

    args = parser.parse_args()
    # Online system identification parameters
    osi_l = 5  # history length for online system identification
    osi_batch_size = 32  # OSI update batch size
    osi_itr = 10         # OSI update iterations
    osi_pretrain_eps = 100  # OSI pretraining episodes, with UP conditioned on true parameters

    # hyper-parameters for RL training
    env_name = ['SawyerReach', 'SawyerPush', 'SawyerSlide'][0]
    max_episodes  = 10000   # 3*12000 for up, 3*10000 for osi
    max_steps   = 80 
    num_workers = 3  # or: mp.cpu_count()
    eval_interval = 100
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
    model_path = '../../../../../data/uposi_td3/model/'+prefix+'_uposi_td3'

    params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
    params_dim = params.shape[0] # dimension of parameters for prediction
    print('Dimension of parameters for prediction: {}'.format(params_dim))
    action_space = env.action_space
    ini_state_space = env.observation_space
    state_space = spaces.Box(-np.inf, np.inf, shape=(ini_state_space.shape[0]+params_dim, ))  # add the dynamics param dim
    if CAT_INTERNAL:
        internal_state_dim = env.get_internal_state_dimension()
        print('Dimension of internal state: ', internal_state_dim)
        _, _, _, info = env.step(np.zeros(action_space.shape[0]))
        internal_action_dim = np.array(info["joint_velocities"]).shape[0]
        print('Dimension of internal action: ', internal_action_dim)
        osi_input_dim = osi_l*(ini_state_space.shape[0]+action_space.shape[0]+internal_state_dim+internal_action_dim)
    else:
        osi_input_dim = osi_l*(ini_state_space.shape[0]+action_space.shape[0])

    td3_trainer=TD3_Trainer(replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
         policy_target_update_interval=policy_target_update_interval, action_range=action_range )
    osi_model = OSINetwork(input_dim = osi_input_dim, output_dim = params_dim)

    if args.up or args.osi: # train universal policy or online system identification model, only one is true
        if args.osi:
            td3_trainer.load_model(model_path)  # load pre-trained universal policy for sequence generation
            osi_model.share_memory()
            osi_loss_list=[]
        else:
            osi_model = None
            td3_trainer.share_memory()

            rewards=[]
            success = []
            eval_rewards = []
            eval_success = []

        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        eval_rewards_queue = mp.Queue()  # used for get offline evaluated rewards from all processes and plot the curve
        success_queue = mp.Queue()  # used for get success events from all processes
        eval_success_queue = mp.Queue()
        osi_loss_queue = mp.Queue()

        processes=[]

        for i in range(num_workers):
            process = Process(target=worker, args=(i, td3_trainer, osi_model, osi_l, osi_input_dim, osi_batch_size, osi_itr, osi_pretrain_eps, env_name, environment_params, environment_wrappers,environment_wrapper_arguments, \
            rewards_queue, eval_rewards_queue, success_queue, eval_success_queue, osi_loss_queue, eval_interval, replay_buffer, max_episodes, max_steps, batch_size, explore_steps, noise_decay,\
            update_itr, explore_noise_scale, eval_noise_scale, reward_scale, DETERMINISTIC, hidden_dim, model_path))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]

        i=0
        while True:  # keep getting the episode reward from the queue
            if args.up:
                # r = rewards_queue.get()
                # succ = success_queue.get()
                eval_r = eval_rewards_queue.get() # this queue has different sample frequence with above two queues, .get() at same time will break the while loop
                eval_succ = eval_success_queue.get() 

                # success.append(succ)
                # rewards.append(r)
                eval_rewards.append(eval_r)
                eval_success.append(eval_succ)
                if len(eval_rewards)%20==0 and len(eval_rewards)>0:
                    # np.save(prefix+'td3_rewards', rewards)
                    # np.save(prefix+'td3_success', success)
                    np.save(prefix+'eval_rewards', eval_rewards)
                    np.save(prefix+'eval_success', eval_success)
            else:
                osi_loss = osi_loss_queue.get()
                osi_loss_list.append(osi_loss)
                np.save('osi_loss', osi_loss_list)
                # writer.add_scalar('Loss/OSI Update', osi_loss, i)
            i+=1

        [p.join() for p in processes]  # finished at the same time
        
    if args.test:
        td3_trainer.load_model(model_path)
        td3_trainer.to_cuda()
        osi_model.load_state_dict(torch.load(model_path+'_osi'))
        env.renderer_on()
        for eps in range(10):
            state =  env.reset()
            env.render()   
            episode_reward = 0
            epi_traj = []
            params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
            params_state = np.concatenate((params, state))

            for step in range(max_steps):
                # using internal state or not
                if CAT_INTERNAL:
                    internal_state = env.get_internal_state()
                    full_state = np.concatenate([state, internal_state])
                else:
                    full_state = state
                action = td3_trainer.policy_net.get_action(params_state, noise_scale=0.0)
                next_state, reward, done, info = env.step(action)
                env.render() 
                if CAT_INTERNAL:
                    target_joint_action = info["joint_velocities"]
                    full_action = np.concatenate([action, target_joint_action])
                else:
                    full_action = action
                epi_traj.append(np.concatenate((full_state, full_action)))

                episode_reward += reward
                if len(epi_traj)>=osi_l:
                    osi_input = stack_data(epi_traj, osi_l)
                    pre_params = osi_model(osi_input).detach().numpy()
                else:
                    pre_params = params
                next_params_state = np.concatenate((pre_params, next_state))
                state=next_state
                params_state = next_params_state

                if done:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
