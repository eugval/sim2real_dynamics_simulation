import numpy as np
from robosuite_extra.env_base import make
import time
import robosuite_extra.utils.transform_utils as T
from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper

import math
from sim2real_policies.utils.logger import Logger
from robosuite_extra.slide_env import SawyerSlide
import pickle
import os
import torch
from sim2real_policies.utils.choose_env import slide_full_randomisation,slide_force_noise_randomisation,slide_force_randomisation
from sim2real_policies.final_policy_testing.network_loading import load, load_model
from sim2real_policies.sys_id.common.utils import query_params
from sim2real_policies.sys_id.universal_policy_online_system_identification.osi import stack_data
from sim2real_policies.final_policy_testing.epi_utils import EPIpolicy_rollout

def main():
    render = True

    # Parameters
    horizon = 60
    total_repeats = 50

    state_dim = 12
    action_dim =  2
    env_name = 'SawyerSlide'

    ### Prepare Environment #####
    env = make(
        'SawyerSlide',
        gripper_type="SlidePanelGripper",
        parameters_to_randomise=slide_full_randomisation,
        randomise_initial_conditions=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        use_indicator_object=False,
        has_renderer=render,
        has_offscreen_renderer=False,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=60,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        pid=True,
    )

    env = FlattenWrapper(GymWrapper(env, ), keys='task-state', add_info=True)
    env._name = env_name
    env.reset()

    for policy_idx in range(1):
        ##### SETTING UP THE POLICY #########
        method = ['Single', 'LSTM', 'EPI', 'UPOSI'][policy_idx]
        if(method == 'Single'):
            alg_idx = 1
        elif(method == 'LSTM'):
            alg_idx = 2
        elif(method == 'UPOSI'):
            alg_idx = 3
            osi_l = 5
            RANDOMISZED_ONLY = True
            DYNAMICS_ONLY = True
            CAT_INTERNAL = False  # sliding env no need for internal state and action
            params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
            params_dim = params.shape[0] # dimension of parameters for prediction
            if CAT_INTERNAL:
                internal_state_dim = env.get_internal_state_dimension()
                _, _, _, info = env.step(np.zeros(action_dim))
                internal_action_dim = np.array(info["joint_velocities"]).shape[0]
                osi_input_dim = osi_l*(state_dim+action_dim+internal_state_dim+internal_action_dim)
            else:
                osi_input_dim = osi_l * (state_dim + action_dim )
            state_dim+=params_dim
        elif(method == 'EPI'):
            alg_idx = 4
            embed_dim = 10
            traj_l = 10
            NO_RESET = True
            embed_input_dim = traj_l*(state_dim+action_dim)
            ori_state_dim = state_dim
            state_dim += embed_dim
        else:
            continue

        alg_name = ['sac', 'td3', 'lstm_td3', 'uposi_td3', 'epi_td3'][alg_idx]
        if method == 'EPI' or method == 'UPOSI':
            randomisation_idx_range = 1
        else:
            randomisation_idx_range = 4

        for randomisation_idx in range(1,2):#randomisation_idx_range
            goal_idx =1
            ###### SETTING UP THE ENVIRONMENT ######
            randomisation_params = [slide_full_randomisation,[] , slide_force_randomisation,
                                    slide_force_noise_randomisation]
            env.change_parameters_to_randomise(randomisation_params[randomisation_idx])
            env.reset()

            if (render):
               # env.viewer.set_camera(camera_id=0)
                env.render()

            # choose which randomisation is applied
            randomisation_type = ['slide_full-randomisation', 'slide_no-randomisation', \
                                  'slide_force-randomisation', 'slide_force-&-noise-randomisation'][
                randomisation_idx]
            number_random_params = [22, 0, 2, 8][randomisation_idx]
            folder_path = '../../../../sawyer/src/sim2real_dynamics_sawyer/assets/rl/'+method +'/' + alg_name + '/model/'
            path = folder_path + env_name + str(
                number_random_params) + '_' + alg_name
            try:
                policy = load(path=path, alg=alg_name, state_dim=state_dim,
                                action_dim=action_dim)
                if method == 'UPOSI':
                    osi_model = load_model(model_name='osi', path=path, input_dim = osi_input_dim, output_dim=params_dim )
                elif method == 'EPI':
                    embed_model = load_model(model_name='embedding', path=path, input_dim = embed_input_dim, output_dim = embed_dim )
                    embed_model.cuda()
                    epi_policy_path = folder_path + env_name + str(number_random_params) + '_' + 'epi_ppo_epi_policy'
                    epi_policy = load(path=epi_policy_path, alg='ppo', state_dim=ori_state_dim, action_dim=action_dim )

            except:
                print(method, randomisation_type)
                continue

            if (alg_name == 'lstm_td3'):
                hidden_out = (torch.zeros([1, 1, 512], dtype=torch.float).cuda(), \
                              torch.zeros([1, 1, 512],
                                          dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                last_action = np.array([0, 0])
            ######################################

            log_save_name = '{}_{}_{}_{}'.format(method, alg_name, randomisation_type, number_random_params)

            for repeat in range(total_repeats):
                # Reset environment
                obs = env.reset()
                if (render):
                   # env.viewer.set_  (camera_id=0)
                    env.render()

                # Setup logger
                log_path = '../../../../data/sliding/sim/{}/goal_{}/trajectory_log_{}.csv'.format(log_save_name,
                                                                                                goal_idx, repeat)

                log_list = ["step", "time",
                            "cmd_j5", "cmd_j6",
                            "obj_x", "obj_y", "obj_z",
                            "sin_z", "cos_z",
                            "obj_vx", "obj_vy", "obj_vz",
                            "a_j5", "a_j6",
                            "v_j5", "v_j6",
                            ]
                logger = Logger(log_list, log_path, verbatim=render)

                i = 0
                mujoco_start_time = env.sim.data.time

                if (alg_name == 'uposi_td3'):
                    uposi_traj = []
                    # params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
                    zero_osi_input = np.zeros(osi_input_dim)
                    pre_params = osi_model(zero_osi_input).detach().numpy()
                    params_state = np.concatenate((pre_params, obs))
                elif (alg_name == 'epi_td3'):


                    if NO_RESET:

                        i = traj_l - 1

                        traj, [last_obs, last_state] = EPIpolicy_rollout(env, epi_policy, obs,
                                                                         mujoco_start_time=mujoco_start_time,
                                                                         logger=logger, data_grabber=None,
                                                                         max_steps=traj_l,
                                                                         params=None)  # only one traj; pass in params to ensure it's not changed
                        state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
                        embedding = embed_model(state_action_in_traj.reshape(-1))
                        embedding = embedding.detach().cpu().numpy()

                        obs = last_obs  # last observation
                        env.set_state(last_state)  # last underlying state
                    else:

                        traj, [last_obs, last_state] = EPIpolicy_rollout(env, epi_policy, obs,
                                                                         mujoco_start_time=mujoco_start_time,
                                                                         logger=None, data_grabber=None,
                                                                         max_steps=traj_l,
                                                                         params=None)  # only one traj; pass in params to ensure it's not changed
                        state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
                        embedding = embed_model(state_action_in_traj.reshape(-1))
                        embedding = embedding.detach().cpu().numpy()


                        params = env.get_dynamics_parameters()
                        env.randomisation_off()
                        env.set_dynamics_parameters(params)  # same as the rollout env
                        obs = env.reset()
                        env.randomisation_on()


                    obs = np.concatenate((obs, embedding))

                while (True):
                    mujoco_elapsed = env.sim.data.time - mujoco_start_time

                    #### CHOOSING THE ACTION #####
                    if (alg_name == 'lstm_td3'):
                        hidden_in = hidden_out
                        action, hidden_out = policy.get_action(obs, last_action, hidden_in, noise_scale=0.0)
                        last_action = action
                    elif (alg_name == 'uposi_td3'):
                        # using internal state or not
                        if CAT_INTERNAL:
                            internal_state = env.get_internal_state()
                            full_state = np.concatenate([obs, internal_state])
                        else:
                            full_state = obs
                        action = policy.get_action(params_state, noise_scale=0.0)
                    else:
                        action = policy.get_action(obs, noise_scale=0.0)
                    ###############################

                    next_obs, reward, done, info = env.step(action)
                    if (alg_name == 'uposi_td3'):
                        if CAT_INTERNAL:
                            target_joint_action = info["joint_velocities"]
                            full_action = np.concatenate([action, target_joint_action])
                        else:
                            full_action = action
                        uposi_traj.append(np.concatenate((full_state, full_action)))

                        if len(uposi_traj)>=osi_l:
                            osi_input = stack_data(uposi_traj, osi_l)
                            pre_params = osi_model(osi_input).detach().numpy()
                        else:
                            zero_osi_input = np.zeros(osi_input_dim)
                            pre_params = osi_model(zero_osi_input).detach().numpy()  

                        next_params_state = np.concatenate((pre_params, next_obs))
                        params_state = next_params_state
                    elif (alg_name == 'epi_td3'):
                        next_obs=np.concatenate((next_obs, embedding))
                    # Grab logging data
                    # assert len(obs) == 12

                    logger.log(i, mujoco_elapsed,
                               action[0], action[1],
                               obs[0], obs[1], obs[2],
                               obs[3], obs[4], obs[5], obs[6],
                               obs[7], obs[8], obs[9],
                               obs[10], obs[11],
                               )

                    obs = next_obs

                    if (render):
                        env.render()

                    i += 1
                    if (i >= horizon or done):
                        break

    env.close()


if __name__ == "__main__":
    main()
