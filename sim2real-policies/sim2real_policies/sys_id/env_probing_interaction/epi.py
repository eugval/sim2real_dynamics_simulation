"""
Environment Probing Interaction (EPI)
https://arxiv.org/abs/1907.11740

Modules:
0. pretrained task-specific policy for collecting transition dataset
1. EPI policy
2. Embedding model
3. EPI prediction model
4. prediction model

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
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from multiprocessing.managers import BaseManager
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
from torch.utils.tensorboard import SummaryWriter
from gym import spaces
from sim2real_policies.utils.policy_networks import DPG_PolicyNetwork, RandomPolicy, PPO_PolicyNetwork
from sim2real_policies.utils.buffers import ReplayBuffer
from sim2real_policies.sys_id.common.utils import query_params, query_key_params, offline_history_collection
from sim2real_policies.sys_id.common.operations import plot, size_splits
from sim2real_policies.sys_id.common.nets import PredictionNetwork,  EmbedNetwork
from sim2real_policies.utils.envs import make_env
from sim2real_policies.utils.evaluate import evaluate, evaluate_epi
from sim2real_policies.ppo.ppo_multiprocess import PPO
from sim2real_policies.td3.td3_multiprocess import TD3_Trainer
from sim2real_policies.utils.load_params import load_params
import pickle
import copy
import time
import argparse
from mujoco_py import MujocoException

# EPI algorthm hyper parameters
DEFAULT_REWARD_SCALE = 1.
PREDICTION_REWARD_SCALE = 0.5
PREDICTION_REWARD_SCALE0 = 1e5
PREDICTOR_LR = 1e-4
EPI_PREDICTOR_LR = 1e-4
EPI_TOTAL_ITR = 1000
EPI_POLICY_ITR = 5  # udpate iterations of EPI polciy; without considering 10 times inside update of ppo policy
PREDICTOR_ITER = 5 #  udpate itertations of predictors and embedding net 
EVAL_INTERVAL = 100 # evaluation interval (episodes) of task-specific policy
HIDDEN_DIM=512
PREFIX=''
NUM_WORKERS = 3
ENV_NAME = ['SawyerReach', 'SawyerPush', 'SawyerSlide'][0]
RANDOMSEED = 2  # random seed
EPI_TRAJ_LENGTH = 10
EPI_EPISODE_LENGTH = 30  # shorter than normal task-specific episode length, only probing through initial exploration steps
EMBEDDING_DIM = 10
DISCRETE = True  # if true, discretized randomization range
VINE = True  # if true, data collected in vine manner
SEPARATION_LOSS = True
NO_RESET = True  # the env is not reset after EPI rollout in each episode
EPI_POLICY_ALG  = ['ppo', 'random'][0]
TASK_POLICY_ALG = ['td3',  'ppo'][0]
# task specific policy
EP_MAX=12000
EP_LEN=100

writer = SummaryWriter()

class EPI(object):
    """
    The class of environment probing interaction policies
    """
    def __init__(self, env, traj_length=EPI_TRAJ_LENGTH, \
        GAMMA=0.9, data_path='./data/', model_path='./model/epi'):
        self.embed_dim = EMBEDDING_DIM
        self.data_path = data_path
        self.model_path = model_path
        self.traj_length = traj_length
        self.GAMMA = GAMMA
        action_space = env.action_space
        state_space = env.observation_space
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.shape[0]
        traj_dim = traj_length*(self.state_dim+self.action_dim)
        state_embedding_space = spaces.Box(-np.inf, np.inf, shape=(state_space.shape[0]+self.embed_dim, ))  # add the embedding param dim
        # initialize epi policy
        if EPI_POLICY_ALG == 'ppo':
            self.epi_policy = PPO(self.state_dim, self.action_dim)
            self.epi_policy.to_cuda()
        elif EPI_POLICY_ALG == 'random':
            self.epi_policy = RandomPolicy(self.action_dim)
        else: 
            raise NotImplementedError
        # initialize task specific policy 
        if TASK_POLICY_ALG == 'ppo':
            self.task_specific_policy = PPO(self.state_dim+self.embed_dim, self.action_dim)
        elif TASK_POLICY_ALG == 'td3':
            self.task_specific_policy = TD3_Trainer(replay_buffer, state_embedding_space, action_space, \
                hidden_dim, q_lr, policy_lr, action_range, policy_target_update_interval)
        else: 
            raise NotImplementedError
        self.embed_net = EmbedNetwork(traj_dim, self.embed_dim, HIDDEN_DIM).cuda()
        self.sa_dim=self.state_dim+self.action_dim
        self.predict_net = PredictionNetwork(self.sa_dim, self.state_dim, HIDDEN_DIM).cuda()
        self.sae_dim = self.sa_dim+self.embed_dim
        self.epi_predict_net = PredictionNetwork(self.sae_dim, self.state_dim, HIDDEN_DIM).cuda()
        
        self.predict_net_optimizer = optim.Adam(self.predict_net.parameters(), PREDICTOR_LR)
        embed_epi_predict_net_params = list(self.epi_predict_net.parameters()) + list(self.embed_net.parameters())
        self.embed_epi_predict_net_optimizer = optim.Adam(embed_epi_predict_net_params, EPI_PREDICTOR_LR)
        self.criterion = nn.MSELoss()

    def save_model(self, model_name=None):
        if model_name is 'predictor_and_embedding':
            torch.save(self.predict_net.state_dict(), self.model_path +'_predictor')
            torch.save(self.epi_predict_net.state_dict(), self.model_path+'_EPIpredictor')
            torch.save(self.embed_net.state_dict(), self.model_path+'_embedding')
            # print('Predictor, EPI Predictor, and Embedding model saved.')
        elif model_name is 'epi_policy':
            self.epi_policy.save_model(path = self.model_path +'_'+EPI_POLICY_ALG+ '_epi_policy')
            # print('EPI policy saved.')
        elif model_name is 'task_specific_policy':
            self.task_specific_policy.save_model(path = self.model_path+'_'+TASK_POLICY_ALG + '_policy')
            # print('Task specific policy saved.')

    def load_model(self, model_name=None):
        if model_name is 'predictor_and_embedding':
            self.predict_net.load_state_dict(torch.load(self.model_path+'_predictor'))
            self.epi_predict_net.load_state_dict(torch.load(self.model_path+'_EPIpredictor'))
            self.embed_net.load_state_dict(torch.load(self.model_path+'_embedding'))
            self.predict_net.eval()
            self.epi_predict_net.eval()
            self.embed_net.eval()
            # print('Predictor, EPI_Predictor, and Embedding model loaded.')
        elif model_name is 'epi_policy':
            self.epi_policy.load_model(path = self.model_path +'_'+EPI_POLICY_ALG+ '_epi_policy')
            # print('EPI policy loaded.')
        elif model_name is 'task_specific_policy':
            self.task_specific_policy.load_model(path =self.model_path+'_'+TASK_POLICY_ALG + '_policy')
            # print('Task specific policy loaded.')
        

    def load_transition_data(self, path = None):
        """
        transition data format: 
        {
        'x_train': (# episodes 1, # steps, state_dim + action_dim)
        'x_test': (# episodes 2, # steps, state_dim + action_dim)
        'y_train': (# episodes 1, # steps, state_dim)
        'y_test': (# episodes 2, # steps, state_dim)
        'param_train': (# episodes 1, # steps, param_dic)
        'param_test': (# episodes 2, # steps, param_dic)
        }
        """
        if path is None:
            path  = self.data_path
            if DISCRETE:
                path+='_discrete'
            if VINE:
                path+='_vine'
            path+='_data.pckl'
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
        data = pickle.load(open(os.path.abspath(path),'rb' ))
        return data

    def where_equal(self, array_of_array, array): 
        """Return index of array_of_array where its item equals to array"""
        idx=[]
        for i in range(array_of_array.shape[0]):
            if (array_of_array[i]==array).all():
                idx.append(i)

        return idx

        """
        original implementation of separation loss in EPI paper is follow:
        def separation_loss(y_true, y_pred):

            y_true = tf.squeeze(y_true)
            env_id, _ = tf.unique(y_true)

            mu = []
            sigma = []
            for i in range(EPI.NUM_OF_ENVS):
                idx = tf.where(tf.equal(y_true, env_id[i]))
                traj = tf.gather(y_pred, idx)
                mu.append(tf.squeeze(K.mean(traj, axis=0)))
                this_sigma = tf.maximum(K.mean(K.std(traj, axis=0))-0.1, 0)
                sigma.append(this_sigma)

            mu = tf.stack(mu)
            r = tf.reduce_sum(mu * mu, 1)
            r = tf.reshape(r, [-1, 1])
            D = (r - 2 * tf.matmul(mu, tf.transpose(mu)) + tf.transpose(r))/tf.constant(EPI.EMBEDDING_DIMENSION, dtype=tf.float32)
            D = tf.sqrt(D + tf.eye(EPI.NUM_OF_ENVS, dtype=tf.float32))
            distance = K.mean(tf.reduce_sum(0.1 - tf.minimum(D, 0.1)))

            sigma = tf.stack(sigma)

            return (distance + K.mean(sigma))*0.01
        """
    def separation_loss(self, params_list, trajs):
        # get list of parameters from ordered dictionary
        list_params = []
        for params in params_list:
            set_params = []
            for key, value in list(params.items()):
                # print(type(value))
                if isinstance(value, np.ndarray):
                    value = value.reshape(-1).astype(float)
                    set_params = np.concatenate((set_params, value)).tolist()
                else:
                    set_params.append(value.astype(float).tolist())
            list_params.append(set_params)
        
        # unique_params_list = np.unique(np.array(list_params).astype('float32'), axis=0)
        unique_params_list = [list(x) for x in set(tuple(x) for x in list_params)]
        number_envs = len(unique_params_list)
        mu=[]
        sigma=[]

        for unique_params in unique_params_list:
            specific_env_idx = self.where_equal(np.array(list_params), np.array(unique_params))
            specific_env_trajs = trajs[specific_env_idx]
            specific_env_trajs = torch.FloatTensor(specific_env_trajs).cuda()
            embedding = self.embed_net(specific_env_trajs) # (len(specific_env_trajs), len(embedding))
            if len(embedding.shape)>2:
                embedding = embedding.view(-1, embedding.shape[-1])
            mu.append(torch.squeeze(torch.mean(embedding, dim=0)))
            this_sigma = torch.max(torch.mean(torch.std(embedding, dim=0))-0.1, 0)[0]  # values of max
            sigma.append(this_sigma)
        mu = torch.stack(mu)
        r = torch.sum(mu*mu, 1)
        r = r.view(-1,1)
        D = (r-2*torch.mm(mu, torch.t(mu)) + torch.t(r))/self.embed_dim
        D = torch.sqrt(D+torch.eye(number_envs).cuda())
        distance = torch.mean(torch.sum(0.1-torch.min(D, torch.as_tensor(0.1).cuda())))
        sigma = torch.stack(sigma)

        return (distance + torch.mean(sigma)) * 0.01

    def predictor_update(self, input, label, trajs, params_list):
        """
        Update the two predictors: 1. predictor with (s,a) as input 2. predictor with (embedding, s,a) as input
        """
        # prediction network update
        state_action_in_trajs = np.array(trajs[:, :, :, :-1]) # remove the rewards
        state_action_in_trajs = state_action_in_trajs.reshape(state_action_in_trajs.shape[0], state_action_in_trajs.shape[1], -1 )# merge last two dims
        input = torch.FloatTensor(input).cuda()
        label = torch.FloatTensor(label).cuda()
        predict = self.predict_net(input)
        predict_loss = self.criterion(predict, label)
        self.predict_net_optimizer.zero_grad()
        predict_loss.backward()
        self.predict_net_optimizer.step()

        batch_loss = []
        # embedding prediction network update
        for epi in range(state_action_in_trajs.shape[0]):
            for traj in range(state_action_in_trajs.shape[1]):
                embedding = self.embed_net(state_action_in_trajs[epi][traj])
                embed_input = torch.cat((embedding.repeat(input[epi].shape[0], 1), input[epi]), dim=1)  # repeat embedding to the batch size
                epi_predict = self.epi_predict_net(embed_input)
                epi_predict_loss = self.criterion(epi_predict, label[epi])
                if torch.isnan(epi_predict_loss).any():  # capture nan cases
                    print('Nan EPI prediction loss')
                    print(state_action_in_trajs[epi][traj], embedding, embed_input, epi_predict, label[epi])
                else:
                    batch_loss.append(epi_predict_loss)
        
        sum_epi_predict_loss = sum(batch_loss)
        if SEPARATION_LOSS:
            separation_loss=self.separation_loss(params_list, state_action_in_trajs)
            if torch.isnan(separation_loss).any():  # capture nan cases
                print('Nan separation loss')
            else:
                sum_epi_predict_loss+=separation_loss
        else:
            separation_loss = 0.
        self.embed_epi_predict_net_optimizer.zero_grad()
        sum_epi_predict_loss.backward()
        self.embed_epi_predict_net_optimizer.step()
        
        return predict_loss, torch.mean(torch.stack(batch_loss)).detach().cpu().numpy(), separation_loss.detach().cpu().numpy()

    def prediction_reward(self, input, label, trajs):
        """  Generate prediction reward for each trajectory """
        r_trajs = []
        sa_trajs = []
        rewards = []
        predict_rewards = []
        state_action_in_trajs = trajs[:, :, :, :-1]  # remove the rewards
        reward_in_trajs = trajs[:, :, :, -1:]  # only the rewards
        state_action_in_trajs_ = state_action_in_trajs.reshape(state_action_in_trajs.shape[0], state_action_in_trajs.shape[1], -1 ) # merge last two dims
        reward_in_trajs_ = reward_in_trajs.reshape(reward_in_trajs.shape[0], reward_in_trajs.shape[1], -1) # merge last two dims
        input = torch.FloatTensor(input).cuda()
        label = torch.FloatTensor(label).cuda()
        for epi in range(state_action_in_trajs_.shape[0]):
            predict = self.predict_net(input[epi])
            predict_loss = self.criterion(predict, label[epi])
            episode_epi_predict_loss = []
            for traj in range(state_action_in_trajs_.shape[1]):
                embedding = self.embed_net(state_action_in_trajs_[epi][traj])
                # print(embedding)
                embed_input = torch.cat((embedding.repeat(input[epi].shape[0], 1), input[epi]), dim=1)
                epi_predict = self.epi_predict_net(embed_input)
                epi_predict_loss = self.criterion(epi_predict, label[epi])  
                predict_r = (epi_predict_loss - predict_loss).detach().cpu().numpy()
                predict_r = np.clip(predict_r*PREDICTION_REWARD_SCALE0, 0, 1000)  # accoring to original implementation, multiplied factor and non-negative
                # print(predict_r)
                augmented_r = DEFAULT_REWARD_SCALE * reward_in_trajs_[epi][traj] + PREDICTION_REWARD_SCALE * predict_r
                rewards.append(augmented_r)
                predict_rewards.append(predict_r)
                r_trajs.append(augmented_r)
                sa_trajs.append(state_action_in_trajs[epi][traj])
        return [np.array(sa_trajs), r_trajs], np.mean(rewards), np.mean(predict_rewards)

    def EPIpolicy_rollout(self, env, max_steps=30, params=None):
        """
        Roll one trajectory with max_steps
        return: 
        traj: shape of (max_steps, state_dim+action_dim+reward_dim)
        s_: next observation
        env.get_state(): next underlying state
        """
        if params is not None: 
            env.set_dynamics_parameters(params)
        # env.renderer_on()
        for _ in range(10): # max iteration for getting a single rollout 
            s = env.reset()
            traj=[]
            for _ in range(max_steps):
                a = self.epi_policy.choose_action(s)
                # env.render()
                try: 
                    s_, r, done, _ = env.step(a)
                except MujocoException:
                    print('EPI Rollout: MujocoException')
                    break
                s_a_r = np.concatenate((s,a, [r]))  # state, action, reward
                traj.append(s_a_r)
                s=s_
            if len(traj) == max_steps:
                break        

        if len(traj)<max_steps:
            print('EPI rollout length smaller than expected!')
            
        return traj, [s_, env.get_state()]
        
    def EPIpolicy_update(self, buffer):
        """ 
        update the EPI policy 
        buffer = [trajectories, rewards]
        trajectories: (# trajs, traj_length, state_dim+action_dim)
        rewards: (# trajs, traj_length)
        """
        # how to learn the policy with the reward for policy, and batch of trajectories to generate one reward
        [trajs, rewards]=buffer
        states = trajs[:, :, :self.state_dim]
        actions = trajs[:, :, self.state_dim:]

        # update ppo
        s_ =  states[:, -1]
        # not considering done here, for reaching no problem for pushing may have problem (done is terminal state)
        v_s_ = self.epi_policy.critic(torch.Tensor([s_]).cuda()).cpu().detach().numpy()[0, 0]  
        discounted_r = []
        for r in np.array(rewards).swapaxes(0,1)[-2::-1]:  # on episode steps dim, [1,2,3,4][-2::-1] gives 3,2,1
            v_s_ = np.expand_dims(r, axis=1) + self.GAMMA * v_s_  # make r has same shape as v_s_: (N, 1)
            discounted_r.append(v_s_)
        discounted_r.reverse()
        # print('r: ', np.array(discounted_r).shape)  # (traj_length, # trajs, 1)
        bs, ba, br = np.vstack(states[:, :-1]), \
            np.vstack(actions[:, :-1]), \
            np.vstack(np.array(discounted_r).swapaxes(0,1))
        # print(bs.shape, ba.shape, br.shape)
        self.epi_policy.update(bs, ba, br)

    def sample_randomized_params(self, dataset_params, batch_size):
        """
        Sample environment parameters from the loaded transition dataset. 
        Note: this implementation is different from original implementation for the paper, 
        but it saves the steps for matching the env idexes between the trajectory generation
        of EPI policy in randomized envs and the envs in transition dataset.
        """
        random_idx = np.random.randint(len(dataset_params), size=batch_size) # with replacement
        return random_idx, np.array(dataset_params)[random_idx]

    def transitions_to_trajs(self, transitions):
        """ 
        Process episodic transition data into trajectories of traj_length.
        Episode length needs to be multiple trajectory length.
        Return: trajectories (np.array), shape: (trans_batch_size, num_trajs_per_episode_transition, traj_length, s_dim+a_dim)
        """
        episode_length = np.array(transitions).shape[1]
        assert episode_length % self.traj_length == 0  # episode length is multiple of trajectory length
        num_trajs_per_episode_transition = int(episode_length/self.traj_length)
        # print("num_trajs_per_episode_transition: ", num_trajs_per_episode_transition)
        split_sizes = num_trajs_per_episode_transition * [self.traj_length]
        trajs = size_splits(torch.Tensor(transitions), split_sizes, dim=1) # split episodic transitions into trajectories, return: tuple
        trajs = np.array([t.numpy() for t in list(trajs)]) # tuple to np.array, (num_trajs_per_episode_transition, trans_batch_size, traj_length, s_dim+a_dim)
        trajs = np.swapaxes(trajs, 0,1) #(trans_batch_size, num_trajs_per_episode_transition, traj_length, s_dim+a_dim)
        return trajs

    def embedding_learn(self, env, epi_episode_length=50, itr=10000, trans_batch_size=20):
        """ 
        Update the embedding network through interatively udpating the predictors and the EPI policy. 
        """
        data = self.load_transition_data()  # pre-collected transition dataset
        print('Data loaded. Training data: {} episodes. Test data: {} episodes.'.format(len(data['x_train']),len(data['x_test'])))
        prediction_loss_list = []
        epi_prediction_loss_list=[]
        separation_loss_list=[]
        overall_reward_list=[]
        prediction_reward_list=[]
        env.randomisation_off()
        for i in range(itr):  # while not converge
            transitions=[]
            # collect transitions with EPI policy
            sampled_index_list, sampled_params_list = self.sample_randomized_params(data['param_train'], trans_batch_size)
            for j in range(trans_batch_size):
                episode_transition, _= self.EPIpolicy_rollout(env, epi_episode_length, sampled_params_list[j])
                if len(episode_transition)< epi_episode_length:
                    episode_transition, _= self.EPIpolicy_rollout(env, epi_episode_length, sampled_params_list[j])
                transitions.append(episode_transition)
            trajs = self.transitions_to_trajs(transitions)

            # update the predictors and the embedding net
            data_x = np.array(data['x_train'])[sampled_index_list]
            data_y = np.array(data['y_train'])[sampled_index_list]
            itr_f_loss = 0
            itr_f_epi_loss = 0
            itr_separation_loss = 0
            for _ in range(PREDICTOR_ITER):
                f_loss, f_epi_loss, separation_loss = self.predictor_update(data_x, data_y, trajs, sampled_params_list)
                itr_f_loss += f_loss
                itr_f_epi_loss += f_epi_loss
                itr_separation_loss += separation_loss
            print('Itr: {}: Predictor loss: {:.5f} | EPI predictor loss: {:.5f} | Separation loss: {:.5f}'\
                .format(i, itr_f_loss, itr_f_epi_loss, itr_separation_loss))
            writer.add_scalar('Loss/Predictor Update', itr_f_loss, i)
            writer.add_scalar('Loss/EPI Predictor Update', itr_f_epi_loss, i)
            writer.add_scalar('Loss/Embedding Separation', itr_separation_loss, i)
            self.save_model(model_name='predictor_and_embedding')
            transitions=[]

            # collect transitions for reward prediction
            sampled_index_list, sampled_params_list = self.sample_randomized_params(data['param_test'], trans_batch_size)
            for j in range(trans_batch_size):
                episode_transition, _ = self.EPIpolicy_rollout(env, epi_episode_length, sampled_params_list[j])
                if len(episode_transition)< epi_episode_length:
                    episode_transition, _= self.EPIpolicy_rollout(env, epi_episode_length, sampled_params_list[j])
                transitions.append(episode_transition)

            trajs = self.transitions_to_trajs(transitions)
            epi_buffer, mean_rewards, mean_predict_rewards = self.prediction_reward(data['x_test'][sampled_index_list], data['y_test'][sampled_index_list], trajs)  # generate reward for each traj
            
            # mean rewards as measures for EPI policy
            writer.add_scalar('Mean Trajectory Reward/EPI policy', mean_rewards, i)
            writer.add_scalar('Mean Trajectory Prediction Reward/EPI policy', mean_predict_rewards, i)

            prediction_loss_list.append(itr_f_loss)
            epi_prediction_loss_list.append(itr_f_epi_loss)
            separation_loss_list.append(itr_separation_loss)
            overall_reward_list.append(mean_rewards)
            prediction_reward_list.append(mean_predict_rewards)
            # update the EPI policy with buffer data of prediction reward
            if EPI_POLICY_ALG != 'random':  # random policy no need for update
                for _ in range(EPI_POLICY_ITR):
                    self.EPIpolicy_update(epi_buffer)

            if i%20 == 0 and i>0:
                self.save_model('epi_policy')
                np.save('prediction_loss', prediction_loss_list)
                np.save('epi_prediction_loss', epi_prediction_loss_list)
                np.save('separation_loss', separation_loss_list)
                np.save('overall_reward', overall_reward_list)
                np.save('prediction_reward', prediction_reward_list)
        env.randomisation_on()

def ppo_worker(id, epi, environment_params, environment_wrappers,environment_wrapper_arguments,\
        eval_rewards_queue, eval_success_queue, batch_size, no_reset):
    """
    learn the task-specific policy with learned embedding network, conditioned on state and embedding;
    general rl training, but use EPI policy to generate trajectory and use embedding net to predict embedding for each episode
    """
    with torch.cuda.device(id % torch.cuda.device_count()):
        # same device
        epi.task_specific_policy.to_cuda()
        epi.epi_policy.to_cuda()
        epi.embed_net.cuda()
        env= make_env('robosuite.'+ENV_NAME, RANDOMSEED, id, environment_params, environment_wrappers, environment_wrapper_arguments)()
        all_ep_r = []
        for ep in range(EP_MAX):
            env.reset()
            params=env.get_dynamics_parameters()
            env.randomisation_off()
            ep_r = 0
            # epi rollout first for each episode
            traj, [last_obs, last_state] = epi.EPIpolicy_rollout(env, max_steps = epi.traj_length, params=params)  # only one traj; pass in params to ensure it's not changed
            state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
            embedding = epi.embed_net(state_action_in_traj.reshape(-1))
            embedding = embedding.detach().cpu().numpy()
            if no_reset:
                s = last_obs  # last observation
                env.set_state(last_state)  # last underlying state
            else:
                env.set_dynamics_parameters(params) # same as the rollout env
                s =  env.reset()

            for t in range(EP_LEN):  # in one episode
                s=np.concatenate((s, embedding))
                a = epi.task_specific_policy.choose_action(s)
                try:
                    s_, r, done, info = env.step(a)
                except MujocoException:
                    print('MujocoException')
                    break
                if info["unstable"]: # capture the case with cube flying away for pushing task
                    break

                epi.task_specific_policy.store_transition(s,a,r)
                s = s_
                s_=np.concatenate((s_, embedding))
                ep_r += r
                # update ppo
                if len(epi.task_specific_policy.state_buffer) == batch_size:
                    epi.task_specific_policy.finish_path(s_, done)
                    epi.task_specific_policy.update()  # update using the buffer's data
                if done:
                    break
            env.randomisation_on()

            epi.task_specific_policy.finish_path(s_, done)
            all_ep_r.append(ep_r)
            if ep%EVAL_INTERVAL==0 and ep>0:
                eval_r, eval_succ = evaluate_epi(env, epi.epi_policy, epi.embed_net, epi.task_specific_policy.actor, epi.traj_length)
                eval_rewards_queue.put(eval_r)
                eval_success_queue.put(eval_succ)   
                epi.save_model('task_specific_policy')
            print('Worker: ', id, '| Episode: ', ep, '| Episode Reward: {:.4f} '.format(ep_r))

        epi.save_model('task_specific_policy')        


def td3_worker(id, epi, environment_params, environment_wrappers, environment_wrapper_arguments, rewards_queue, eval_rewards_queue, success_queue,\
            eval_success_queue, replay_buffer, batch_size, explore_steps, noise_decay, update_itr, explore_noise_scale, \
            eval_noise_scale, reward_scale, DETERMINISTIC, hidden_dim, no_reset):
    '''
    the function for sampling with multi-processing
    '''

    with torch.cuda.device(id % torch.cuda.device_count()):
        # same device
        epi.task_specific_policy.to_cuda()
        epi.epi_policy.to_cuda()
        epi.embed_net.cuda()
        print(epi.task_specific_policy, replay_buffer)
        env= make_env('robosuite.'+ENV_NAME, RANDOMSEED, id, environment_params, environment_wrappers,environment_wrapper_arguments)()
        action_dim = env.action_space.shape[0]
        frame_idx=0
        rewards=[]
        current_explore_noise_scale = explore_noise_scale
        # training loop
        for eps in range(EP_MAX):
            env.reset()
            params=env.get_dynamics_parameters()
            env.randomisation_off()
            # epi rollout first for each episode
            traj, [last_obs, last_state] = epi.EPIpolicy_rollout(env, max_steps = epi.traj_length, params=params)  # only one traj; pass in params to ensure it's not changed
            state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
            embedding = epi.embed_net(state_action_in_traj.reshape(-1))
            embedding = embedding.detach().cpu().numpy()

            episode_reward = 0
            if no_reset:
                state = last_obs  # last observation
                env.set_state(last_state)  # last underlying state
            else:
                env.set_dynamics_parameters(params) # same as the rollout env
                state =  env.reset()

            current_explore_noise_scale = current_explore_noise_scale*noise_decay
            state=np.concatenate((state, embedding))
            for step in range(EP_LEN):
                if frame_idx > explore_steps:
                    action = epi.task_specific_policy.policy_net.get_action(state, noise_scale=current_explore_noise_scale)
                else:
                    action = epi.task_specific_policy.policy_net.sample_action()
        
                try:
                    next_state, reward, done, info = env.step(action)
                    if environment_params["has_renderer"] and environment_params["render_visual_mesh"]:
                        env.render()   
                except KeyboardInterrupt:
                    print('Finished')
                    epi.save_model('task_specific_policy')    
                except MujocoException:
                    print('Task specific policy: MujocoException')
                    break

                if info["unstable"]: # capture the case with cube flying away for pushing task
                    break
                next_state=np.concatenate((next_state, embedding))
                replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                frame_idx += 1
                
                # if len(replay_buffer) > batch_size:
                if replay_buffer.get_length() > batch_size:
                    for i in range(update_itr):
                        _=epi.task_specific_policy.update(batch_size, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                
                if done:
                    break
            env.randomisation_on()
            print('Worker: ', id, '| Episode: ', eps, '| Episode Reward: ', episode_reward)
            rewards_queue.put(episode_reward)
            success_queue.put(info['success'])

            if eps % EVAL_INTERVAL == 0 and eps>0:
                # plot(rewards, id)
                epi.save_model('task_specific_policy')
                eval_r, eval_succ = evaluate_epi(env, epi.epi_policy, epi.embed_net, epi.task_specific_policy.policy_net, epi.traj_length)
                eval_rewards_queue.put(eval_r)
                eval_success_queue.put(eval_succ)

        epi.save_model('task_specific_policy')        

def specific_policy_learn(epi, environment_params, environment_wrappers, environment_wrapper_arguments, no_reset=True):
    """ 
    multi-process for learning the task-specific policy rather than
    using the single-process in epi class
    """
    epi.load_model('predictor_and_embedding')
    epi.load_model('epi_policy')
    epi.task_specific_policy.share_memory()
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
        if TASK_POLICY_ALG == 'ppo':
            process = Process(target=ppo_worker, args=(i, epi, environment_params, environment_wrappers, \
            environment_wrapper_arguments, eval_rewards_queue, eval_success_queue, batch_size, no_reset))  # the args contain shared and not shared
        elif TASK_POLICY_ALG == 'td3':
            process = Process(target=td3_worker, args=(i, epi, environment_params, environment_wrappers,\
            environment_wrapper_arguments, rewards_queue, eval_rewards_queue, success_queue, eval_success_queue,\
            replay_buffer, batch_size, explore_steps, noise_decay,\
            update_itr, explore_noise_scale, eval_noise_scale, reward_scale, DETERMINISTIC, hidden_dim, no_reset))
        else: 
            raise NotImplementedError
        process.daemon=True  # all processes closed when the main stops
        processes.append(process)

    [p.start() for p in processes]
    while True:  # keep geting the episode reward from the queue
        eval_r = eval_rewards_queue.get() 
        eval_succ = eval_success_queue.get() 

        eval_rewards.append(eval_r)
        eval_success.append(eval_succ)

        if len(eval_rewards)%20==0 and len(eval_rewards)>0:
            np.save(PREFIX+'eval_rewards', eval_rewards)
            np.save(PREFIX+'eval_success', eval_success)

    [p.join() for p in processes]  # finished at the same time

def test(env, no_reset):
    epi.load_model('predictor_and_embedding')
    epi.load_model('epi_policy')
    epi.load_model('task_specific_policy')
    epi.task_specific_policy.to_cuda()
    epi.epi_policy.to_cuda()
    epi.embed_net.cuda()
    action_dim = env.action_space.shape[0]
    env.renderer_on()
    for eps in range(10):
        env.reset()
        params=env.get_dynamics_parameters()
        env.randomisation_off()
        # epi rollout first for each episode
        traj, [last_obs, last_state] = epi.EPIpolicy_rollout(env, max_steps = epi.traj_length, params=params)  # only one traj; pass in params to ensure it's not changed
        state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
        embedding = epi.embed_net(state_action_in_traj.reshape(-1))
        embedding = embedding.detach().cpu().numpy()
        episode_reward = 0
        if no_reset:
            state = last_obs  # last observation
            env.set_state(last_state)  # last underlying state
        else:
            env.set_dynamics_parameters(params) # same as the rollout env
            state =  env.reset()
        state=np.concatenate((state, embedding))
        for step in range(EP_LEN):
            action = epi.task_specific_policy.policy_net.get_action(state, noise_scale=0.0)
            next_state, reward, done, info = env.step(action)
            env.render() 
            next_state=np.concatenate((next_state, embedding))
            state = next_state
            episode_reward += reward      
            if done:
                break
        env.randomisation_on()
        print('Worker: ', id, '| Episode: ', eps, '| Episode Reward: ', episode_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EPI.')
    parser.add_argument('--epi', dest='epi', action='store_true', default=False)
    parser.add_argument('--task', dest='task', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)

    args = parser.parse_args()

    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(ENV_NAME)
    prefix=ENV_NAME+str(len(environment_params["parameters_to_randomise"]))  # number of randomised parameters
    model_path = '../../../../../data/epi/model/'+prefix+'_epi'
    if TASK_POLICY_ALG =='td3': 
        # load td3 hyper parameters
        [action_range, batch_size, explore_steps, update_itr, explore_noise_scale, eval_noise_scale, reward_scale, \
        hidden_dim, noise_decay, policy_target_update_interval, q_lr, policy_lr, replay_buffer_size, DETERMINISTIC] = \
            load_params('td3', ['action_range', 'batch_size', 'explore_steps', 'update_itr', 'explore_noise_scale',\
            'eval_noise_scale', 'reward_scale', 'hidden_dim', 'noise_decay', \
                'policy_target_update_interval', 'q_lr', 'policy_lr','replay_buffer_size', 'deterministic'] )
        
        # load replay buffer when off-policy
        BaseManager.register('ReplayBuffer', ReplayBuffer)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer(replay_buffer_size)  # share the replay buffer through manager
    elif TASK_POLICY_ALG =='ppo':
        [batch_size] = load_params('ppo', ['batch_size'])

    epi = EPI(env, data_path='./data/'+ENV_NAME, model_path = model_path)
    
    if args.epi:
        epi.embedding_learn(env, epi_episode_length=EPI_EPISODE_LENGTH, itr = EPI_TOTAL_ITR)
    elif args.task:
        specific_policy_learn(epi,  environment_params, environment_wrappers, environment_wrapper_arguments, no_reset=NO_RESET)
    elif args.test:
        test(env, no_reset = NO_RESET)
    else:   
        pass
