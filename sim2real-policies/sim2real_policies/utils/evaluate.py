import numpy as np
import torch
from sim2real_policies.sys_id.common.utils import query_params
from mujoco_py import MujocoException

def testing_envionments_generator_random(env, number_of_envs):
    rng = np.random.RandomState()
    rng.seed(3)

    def log_uniform(low=1e-10, high=1., size=None):
        return np.exp(rng.uniform(np.log(low), np.log(high), size))

    def ranged_random_choice(low, high, size=1):
        vals = np.arange(low,high+1)
        return rng.choice(vals, size)

    def select_appropriate_distribution(key):
        if (
                key == 'joint_forces'
                or key == 'acceleration_forces'
                or key == 'eef_forces'
                or key == 'obj_forces'

                or key == 'timestep_parameter'
                or key == 'pid_iteration_time'
                or key == 'mujoco_timestep'

                or key == 'action_additive_noise'
                or key == 'action_multiplicative_noise'
                or key == 'action_systematic_noise'

                or key == 'eef_obs_position_noise'
                or key == 'eef_obs_velocity_noise'
                or key == 'obj_obs_position_noise'
                or key == 'obj_obs_velocity_noise'
                or key == 'obj_angle_noise'

                or key == 'link_masses'

                or key == 'obj_density'
                or key == 'obj_sliding_friction'
            ):
            return rng.uniform
        elif (
                key == 'eef_timedelay'
                or key == 'obj_timedelay'
        ):
            return ranged_random_choice
        else:
            return log_uniform

    factors_for_randomisation = env.get_factors_for_randomisation()
    parameter_ranges = env.get_parameter_sampling_ranges()
    params_list = []
    for _ in range(0, number_of_envs):
        params = {}

        randomised_params = env.get_randomised_parameters()
        for param_key in randomised_params :
            parameter_range = parameter_ranges[param_key]

            if (parameter_range.shape[0] == 1):
                params[param_key]= np.asarray(parameter_range[0])
            elif (parameter_range.shape[0] == 2):
                distribution = select_appropriate_distribution(param_key)
                size = env.default_dynamics_parameters[param_key].shape
                params[param_key]=np.asarray(
                    factors_for_randomisation[param_key] * distribution(*parameter_ranges[param_key], size=size))
            else:
                raise RuntimeError('Parameter radomisation range needs to be of shape {1,2}xN')
        params_list.append(params)
    return params_list


def evaluate(env, policy, up=False, eval_eipsodes=5, Projection=False, proj_net=None, num_envs=5, randomised_only=False, dynamics_only=False):
    '''
    Evaluate the policy during training time with fixed dynamics
    :param: eval_episodes: evaluating episodes for each env
    :param: up (bool): universal policy conditioned on dynamics parameters
    '''
    number_of_episodes = eval_eipsodes
    return_per_test_environment = []
    success_per_test_environment = []
    # initial_dynamics_params = env.get_dynamics_parameters()
    dynamics_params = testing_envionments_generator_random(env, num_envs) # before randomisation off
    ### SUGGESTED CHANGE ####
    initial_dynamics_params, _ = env.randomisation_off()
    ###################

    # do visualization
    # print('Starting testing')
    for params in dynamics_params:
        mean_return = 0.0
        success = 0

        # print('with parameters {} ...'.format(params))
        for i in range(number_of_episodes):
            #### SUGGESTED CHANGE ####
            # First set the parameters and then reset for them to take effect.
            env.set_dynamics_parameters(params)
            obs = env.reset()
            p = env.get_dynamics_parameters()
            #######
            # obs = env.reset()
            # env.set_dynamics_parameters(params)  # set parameters after reset

            if up:
                env.randomisation_on() # open randosimation to query params correctly
                params_list = query_params(env, randomised_only=randomised_only, dynamics_only=dynamics_only)
                env.randomisation_off()
                if Projection and proj_net is not None:
                    params_list = proj_net.get_context(params_list)
                obs = np.concatenate((params_list, obs))

            episode_return = 0.0
            done = False
            while not done:
                action = policy.get_action(obs)
                obs, reward, done, info = env.step(action)
                episode_return += reward

                if up:
                    obs = np.concatenate((params_list, obs))
                
                if (done):
                    mean_return += episode_return
                    success += int(info['success'])
                    break

        mean_return /= number_of_episodes
        success_rate = float(success) / float(number_of_episodes)
        # print('the mean return is {}'.format(mean_return))
        return_per_test_environment.append(mean_return)
        success_per_test_environment.append(success_rate)

    #### SUGGESTED CHANGE ####
    # Only need to call randomisation_on, this will restore the previous parameters and the
    # randoisation too. Note that this will only take effect in the next reset, where parameters
    # are going to be randomised again
    env.randomisation_on()
    #####

    # env.set_dynamics_parameters(initial_dynamics_params)  # for non-randomisation cases, need to set original dynamics parameters
    print('total_average_return_over_test_environments : {}'.format(np.mean(return_per_test_environment)))
    return np.mean(return_per_test_environment), np.mean(success_per_test_environment)

def evaluate_epi(env, epi_policy, embed_net, task_policy, traj_length, eval_eipsodes=5, num_envs=5):
    """
    Apart from evaluate(), epi policy conditions on the embedding of trajectory
    :param: epi_policy: the policy for rollout a trajectory
    :param: embed_net: embedding network of the trajectory
    :param: task_policy: the policy for evaluation, conditioned on the embedding
    :param: traj_length: length of trajectory
    """
    number_of_episodes = eval_eipsodes
    return_per_test_environment = []
    success_per_test_environment = []
    dynamics_params = testing_envionments_generator_random(env, num_envs)
    # initial_dynamics_params = env.get_dynamics_parameters()
    initial_dynamics_params, _ = env.randomisation_off()

    def policy_rollout(env, policy, max_steps=30, params=None):
        """
        Roll one trajectory with max_steps
        return: traj, shape of (max_steps, state_dim+action_dim+reward_dim)
        """
        # s = env.reset()
        # if params is not None:
        #     env.set_dynamics_parameters(params)  # make sure .reset() not change env params

        if params is not None:
            env.set_dynamics_parameters(params)  # make sure .reset() not change env params
        for _ in range(3):
            s = env.reset()
            traj=[]
            for _ in range(max_steps):
                a = policy.choose_action(s)
                try:
                    s_, r, done, _ = env.step(a)
                except MujocoException:
                    print('MujocoException')
                    break
                s_a_r = np.concatenate((s,a, [r]))  # state, action, reward
                traj.append(s_a_r)
                s=s_
            if len(traj) == max_steps:
                break        

        if len(traj)<max_steps:
            print('EPI rollout length smaller than expected!')
        return traj

    # do visualization
    # print('Starting testing')
    for params in dynamics_params:
        mean_return = 0.0
        success = 0

        # print('with parameters {} ...'.format(params))
        for i in range(number_of_episodes):
            traj = policy_rollout(env, epi_policy, max_steps = traj_length, params=params)  # only one traj
            state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
            embedding = embed_net(state_action_in_traj.reshape(-1))
            embedding = embedding.detach().cpu().numpy()
            # s = env.reset()
            # env.set_dynamics_parameters(params)  # set parameters after reset
            env.set_dynamics_parameters(params)
            s = env.reset()

            ep_r = 0.0
            done = False
            while not done:
                s=np.concatenate((s, embedding))
                a = task_policy.get_action(s)
                s_, r, done, info = env.step(a)
                s = s_
                ep_r += r
            
                if (done):
                    mean_return += ep_r
                    success += int(info['success'])
                    break

        mean_return /= number_of_episodes
        success_rate = float(success) / float(number_of_episodes)
        # print('the mean return is {}'.format(mean_return))
        return_per_test_environment.append(mean_return)
        success_per_test_environment.append(success_rate)

    env.randomisation_on()

    # env.set_dynamics_parameters(initial_dynamics_params)  # set original dynamics parameters
    print('total_average_return_over_test_environments : {}'.format(np.mean(return_per_test_environment)))
    return np.mean(return_per_test_environment), np.mean(success_per_test_environment)



def evaluate_lstm(env, policy, hidden_dim, eval_eipsodes=5, num_envs=5):
    """
    Evaluate the policy during training time with fixed dynamics
    :param: eval_episodes: evaluating episodes for each env
    """
    number_of_episodes = eval_eipsodes
    return_per_test_environment = []
    success_per_test_environment = []
    dynamics_params = testing_envionments_generator_random(env, num_envs)
    # initial_dynamics_params = env.get_dynamics_parameters()
    initial_dynamics_params, _ = env.randomisation_off()


    # do visualization
    # print('Starting testing')
    for params in dynamics_params:
        mean_return = 0.0
        success = 0

        # print('with parameters {} ...'.format(params))
        for i in range(number_of_episodes):
            env.set_dynamics_parameters(params)
            obs = env.reset()
            
            # obs = env.reset()
            # env.set_dynamics_parameters(params)

            episode_return = 0.0
            done = False
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            last_action = env.action_space.sample()
            while not done:
                hidden_in = hidden_out
                action, hidden_out= policy.get_action(obs, last_action, hidden_in)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                last_action = action
                if (done):
                    mean_return += episode_return
                    success += int(info['success'])
                    break

        mean_return /= number_of_episodes
        success_rate = float(success) / float(number_of_episodes)
        # print('the mean return is {}'.format(mean_return))
        return_per_test_environment.append(mean_return)
        success_per_test_environment.append(success_rate)

    env.randomisation_on()

    # env.set_dynamics_parameters(initial_dynamics_params)  # set original dynamics parameters
    print('total_average_return_over_test_environments : {}'.format(np.mean(return_per_test_environment)))
    return np.mean(return_per_test_environment), np.mean(success_per_test_environment)
