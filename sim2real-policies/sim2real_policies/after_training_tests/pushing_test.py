import numpy as np
from robosuite_extra.env_base import make
import time
import robosuite_extra.utils.transform_utils as T
from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper

import math
from sim2real_policies.utils.logger import Logger
from robosuite_extra.push_env import SawyerPush
import pickle
import os
import torch
from sim2real_policies.utils.choose_env import push_force_noise_randomisation,push_force_randomisation,push_full_randomisation
from sim2real_policies.final_policy_testing.network_loading import load, load_model
from mujoco_py import MujocoException
from sim2real_policies.sys_id.common.utils import query_params
from sim2real_policies.sys_id.universal_policy_online_system_identification.osi import stack_data
from .epi_utils import EPIpolicy_rollout

PUSH_GOALS = [np.array([3e-3,15.15e-2]),np.array([9.5e-2,11.01e-2]), np.array([-11.5e-2,17.2e-2])]


def grab_data(info, world_pose_in_base):
    # Grab the data
    goal_pos_in_world = np.concatenate([info['goal_pos_in_world'], [1.0]])

    goal_pos_in_base = world_pose_in_base.dot(goal_pos_in_world)
    goal_pos_in_base = goal_pos_in_base / goal_pos_in_base[3]

    gripper_pos_in_world = np.concatenate([info['eef_pos_in_world'], [1.0]])
    gripper_vel_in_world = info['eef_vel_in_world']

    gripper_pos_in_base = world_pose_in_base.dot(gripper_pos_in_world)
    gripper_pos_in_base = gripper_pos_in_base / gripper_pos_in_base[3]
    gripper_vel_in_base = world_pose_in_base[:3, :3].dot(gripper_vel_in_world)

    object_pos_in_world = np.concatenate([info['object_pos_in_world'], [1.0]])
    object_vel_in_world = info['object_vel_in_world']
    z_angle = info['z_angle']

    object_pos_in_base = world_pose_in_base.dot(object_pos_in_world)
    object_pos_in_base = object_pos_in_base / object_pos_in_base[3]
    object_vel_in_base = world_pose_in_base[:3, :3].dot(object_vel_in_world)


    return goal_pos_in_base, gripper_pos_in_base, gripper_vel_in_base, object_pos_in_base, object_vel_in_base, z_angle,


def main():
    render = True

    # Parameters
    horizon = 80
    total_repeats = 50

    state_dim = 10
    action_dim = 2
    env_name = 'SawyerPush'

    ### Prepare Environment #####
    env = make(
        'SawyerPush',
        gripper_type="PushingGripper",
        parameters_to_randomise=push_full_randomisation,
        randomise_initial_conditions=False,
        table_full_size=(0.8, 1.6, 0.719),
        table_friction=(0.001, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=render,
        has_offscreen_renderer=False,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10.,
        horizon=200,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        pid=True,
    )

    env = FlattenWrapper(GymWrapper(EEFXVelocityControl(env, dof=2,max_action=0.1, ),),keys='task-state',add_info=True)
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
            CAT_INTERNAL = True
            params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
            params_dim = params.shape[0] # dimension of parameters for prediction
            if CAT_INTERNAL:
                internal_state_dim = env.get_internal_state_dimension()
                _, _, _, info = env.step(np.zeros(action_dim))
                internal_action_dim = np.array(info["joint_velocities"]).shape[0]
                osi_input_dim = osi_l*(state_dim+action_dim+internal_state_dim+internal_action_dim)
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

        for randomisation_idx in range(1,2): #randomisation_idx_range
            for goal_idx, goal_pos in enumerate(PUSH_GOALS):
                goal_idx= goal_idx+1
                ###### SETTING UP THE ENVIRONMENT ######
                randomisation_params = [push_full_randomisation, [], push_force_randomisation,
                                        push_force_noise_randomisation]
                env.change_parameters_to_randomise(randomisation_params[randomisation_idx])

                start_pos = np.array([0., -16.75e-2])
                env._set_gripper_neutral_offset(*start_pos)
                env._set_goal_neutral_offset(*goal_pos)
                env.reset()

                if (render):
                    env.viewer.set_camera(camera_id=0)
                    env.render()

                base_pos_in_world = env.sim.data.get_body_xpos("base")
                base_rot_in_world = env.sim.data.get_body_xmat("base").reshape((3, 3))
                base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
                world_pose_in_base = T.pose_inv(base_pose_in_world)

                # choose which randomisation is applied
                randomisation_type = ['push_full-randomisation', 'push_no-randomisation', \
                                      'push_force-randomisation', 'push_force-&-noise-randomisation'][
                    randomisation_idx]
                number_random_params = [23, 0, 2, 9][randomisation_idx]
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
                    last_action = np.array([0, 0, 0])
                ######################################

                log_save_name = '{}_{}_{}_{}'.format(method, alg_name, randomisation_type, number_random_params)

                for repeat in range(total_repeats):
                    # Reset environment
                    obs = env.reset()
                    if (render):
                        env.viewer.set_camera(camera_id=0)
                        env.render()

                    # Establish extra frame transforms
                    base_rot_in_eef = env.init_right_hand_orn.T

                    # Setup logger
                    log_path = '../../../../data/pushing/sim/{}/goal_{}/trajectory_log_{}.csv'.format(log_save_name,
                                                                                                    goal_idx, repeat)

                    log_list = ["step", "time",
                                "cmd_eef_vx", "cmd_eef_vy",
                                "goal_x", "goal_y", "goal_z",
                                "eef_x", "eef_y", "eef_z",
                                "eef_vx", "eef_vy", "eef_vz",
                                "object_x", "object_y", "object_z",
                                "object_vx", "object_vy", "object_vz",
                                "z_angle",
                                "obs_0", "obs_1", "obs_2",
                                "obs_3", "obs_4", "obs_5",
                                "obs_6", "obs_7","obs_8", "obs_9"]

                    logger = Logger(log_list, log_path, verbatim=render)

                    i = 0
                    mujoco_start_time = env.sim.data.time

                    if (alg_name == 'uposi_td3'):
                        uposi_traj = []
                        zero_osi_input = np.zeros(osi_input_dim)
                        pre_params = osi_model(zero_osi_input).detach().numpy()
                        params_state = np.concatenate((pre_params, obs))
                    elif (alg_name == 'epi_td3'):
                        if NO_RESET:

                            i=traj_l-1

                            traj, [last_obs, last_state] = EPIpolicy_rollout(env, epi_policy, obs,
                                                                             mujoco_start_time=mujoco_start_time,
                                                                             logger=logger, data_grabber=grab_data,
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
                            env.set_dynamics_parameters(params) # same as the rollout env
                            obs =  env.reset() # Reset fot the params to take effect on the reset environment
                            env.randomisation_on()


                        obs=np.concatenate((obs, embedding))


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
                        try:
                            next_obs, reward, done, info = env.step(action)
                        except MujocoException:
                            print('mujoco exception in iteration {} of {} {}'.format(i, method, randomisation_type))
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
                        goal_pos_in_base, eef_pos_in_base, eef_vel_in_base, \
                        object_pos_in_base, object_vel_in_base, z_angle, = grab_data(info, world_pose_in_base)

                        action_3d = np.concatenate([action, [0.0]])
                        action_3d_in_base = base_rot_in_eef.dot(action_3d)

                        logger.log(i, mujoco_elapsed,
                                   action_3d_in_base[0], action_3d_in_base[1],
                                   goal_pos_in_base[0], goal_pos_in_base[1], goal_pos_in_base[2],
                                   eef_pos_in_base[0], eef_pos_in_base[1], eef_pos_in_base[2],
                                   eef_vel_in_base[0], eef_vel_in_base[1], eef_vel_in_base[2],
                                   object_pos_in_base[0], object_pos_in_base[1], object_pos_in_base[2],
                                   object_vel_in_base[0], object_vel_in_base[1], object_vel_in_base[2],
                                   z_angle[0],
                                   obs[0], obs[1], obs[2],
                                   obs[3], obs[4], obs[5],
                                   obs[6], obs[7], obs[8],
                                   obs[9],
                                   )

                        obs = next_obs

                        if (render):
                            env.render()
                        i += 1

                        if (i >= horizon):
                            break

    env.close()


if __name__ == "__main__":
    main()
