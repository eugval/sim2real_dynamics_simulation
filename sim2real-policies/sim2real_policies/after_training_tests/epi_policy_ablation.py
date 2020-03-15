import numpy as np
from robosuite_extra.env_base import make
import time
import robosuite_extra.utils.transform_utils as T
from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper
from sim2real_policies.final_policy_testing.network_loading import load, load_model
import math
from sim2real_calibration_characterisation.utils.logger import Logger
from robosuite_extra.reach_env import SawyerReach
import pickle
import os
import torch
from sim2real_policies.utils.choose_env import reach_full_randomisation, push_full_randomisation, slide_full_randomisation
from sim2real_policies.sys_id.common.utils import query_params
from sim2real_policies.sys_id.universal_policy_online_system_identification.osi import stack_data
from sim2real_policies.final_policy_testing.epi_utils import EPIpolicy_rollout


from mujoco_py import MujocoException

REACH_GOALS= [ np.array([0.,0.]),np.array([-4.465e-2,5.85e-2]),np.array([8.37e-2,-5.78e-2])]
PUSH_GOALS = [np.array([3e-3,15.15e-2]),np.array([9.5e-2,11.01e-2]), np.array([-11.5e-2,17.2e-2])]
SLIDE_GOALS = [ np.array([0.,0.])]

def grab_reach_data(info, world_pose_in_base):
    # Grab the data
    eef_pos_in_world = info['eef_pos_in_world']
    eef_vel_in_world = info['eef_vel_in_world']
    goal_pos_in_world = info['goal_pos_in_world']

    eef_pos_in_world = np.concatenate([eef_pos_in_world, [1.0]])
    goal_pos_in_world = np.concatenate([goal_pos_in_world, [1.0]])

    eef_pos_in_base = world_pose_in_base.dot(eef_pos_in_world)
    eef_pos_in_base = eef_pos_in_base / eef_pos_in_base[3]
    eef_vel_in_base = world_pose_in_base[:3, :3].dot(eef_vel_in_world)

    goal_pos_in_base = world_pose_in_base.dot(goal_pos_in_world)
    goal_pos_in_base = goal_pos_in_base /  goal_pos_in_base[3]


    return eef_pos_in_base, eef_vel_in_base, goal_pos_in_base



def grab_push_data(info, world_pose_in_base):
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





def make_env(task,render):
    if(task == 'reaching'):
    ### Prepare Environment #####
        env = make(
            'SawyerReach',
            gripper_type="PushingGripper",
            parameters_to_randomise=reach_full_randomisation,
            randomise_initial_conditions=False,
            table_full_size=(0.8, 1.6, 0.719),
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            use_indicator_object=False,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10,
            horizon=100,
            ignore_done=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            pid=True,
            success_radius=0.01
        )

        env = FlattenWrapper(GymWrapper(EEFXVelocityControl(env, dof=3, max_action=0.1), ), keys='task-state',
                             add_info=True)

        env._name ='SawyerReach'

    elif(task == 'pushing'):
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
            gripper_visualization=True,
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
        env = FlattenWrapper(GymWrapper(EEFXVelocityControl(env, dof=2, max_action=0.1, ), ), keys='task-state',
                             add_info=True)
        env._name = 'SawyerPush'

    elif(task == 'sliding'):
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
        env._name = 'SawyerSlide'
    return env


def main():
    render = False

    # Parameters
    total_repeats = 50


    for env_name in [ 'SawyerReach', 'SawyerPush' ]: #, 'SawyerSlide',
        for use_white_noise in [  True, False]: # True
            if(env_name == 'SawyerReach'):
                task = 'reaching'
            elif(env_name == 'SawyerPush'):
                task = 'pushing'
            elif(env_name == 'SawyerSlide'):
                task = 'sliding'

            env = make_env(task,render)
            env.reset()



            if(task == 'reaching'):
                GOALS = REACH_GOALS
            elif(task == 'pushing'):
                GOALS = PUSH_GOALS
            elif(task == 'sliding'):
                GOALS = SLIDE_GOALS

            for goal_idx, goal_pos in enumerate(GOALS):

                if (env_name == 'SawyerReach'):
                    state_dim = 6
                    action_dim = 3
                    horizon = 50
                elif (env_name == 'SawyerPush'):
                    horizon = 80
                    state_dim = 10
                    action_dim = 2
                elif (env_name == 'SawyerSlide'):
                    horizon = 60
                    state_dim = 12
                    action_dim = 2


                goal_idx = goal_idx+1
                #### setting up the environment ######
                if(task == 'reaching'):
                    env._set_goal_neutral_offset(*goal_pos)
                elif(task == 'pushing'):
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


                randomisation_type = 'reach_full-randomisation'
                if (task == 'reaching'):
                    number_random_params = 14
                elif (task == 'pushing'):
                    number_random_params = 23
                elif (task == 'sliding'):
                    number_random_params = 22



                ############# SETTING UP THE POLICY  ###########

                method = 'EPI'
                alg_name =  'epi_td3'

                embed_dim = 10
                traj_l = 10
                NO_RESET = True
                embed_input_dim = traj_l*(state_dim+action_dim)
                ori_state_dim = state_dim
                state_dim += embed_dim

                folder_path = '../../../../sawyer/src/sim2real_dynamics_sawyer/assets/rl/'+method +'/' + alg_name + '/model/'
                path = folder_path + env_name + str(
                    number_random_params) + '_' + alg_name

                print(embed_input_dim)
                print(embed_dim)

                embed_model = load_model(model_name='embedding', path=path, input_dim = embed_input_dim, output_dim = embed_dim )
                embed_model.cuda()
                epi_policy_path = folder_path + env_name + str(number_random_params) + '_' + 'epi_ppo_epi_policy'
                epi_policy = load(path=epi_policy_path, alg='ppo', state_dim=ori_state_dim, action_dim=action_dim )

                policy = load(path=path, alg=alg_name, state_dim=state_dim,
                              action_dim=action_dim)

                ###############################################



                log_save_name = '{}_{}_{}_{}'.format(method, alg_name, randomisation_type, number_random_params)

                for repeat in range(total_repeats):
                    #Reset environment
                    obs = env.reset()

                    #Establish extra frame transforms
                    if(task != 'sliding'):
                        base_rot_in_eef = env.init_right_hand_orn.T

                    #Setup logger
                    if(use_white_noise):
                        noise_folder = 'noise'
                    else:
                        noise_folder = 'normal'

                    log_path = '../../../../data/epi_ablation/{}/{}/{}/goal_{}/trajectory_log_{}.csv'.format(task, noise_folder,  log_save_name,goal_idx,repeat)

                    if (task == 'reaching'):
                        log_list = ["step", "time",
                                    "cmd_eef_vx", "cmd_eef_vy", "cmd_eef_vz",
                                    "eef_x", "eef_y", "eef_z",
                                    "eef_vx", "eef_vy", "eef_vz",
                                    "goal_x", "goal_y", "goal_z",
                                    "obs_0", "obs_1", "obs_2",
                                    "obs_3", "obs_4", "obs_5"
                                    ]

                    elif (task == 'pushing'):
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
                                    "obs_6", "obs_7", "obs_8", "obs_9"]
                    elif (task == 'sliding'):
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

                    if (use_white_noise):
                        embedding = np.random.uniform(-1., 1., size=(embed_dim,))
                    else:
                        # TODO: Check this change -> taking out the reset in EPIpolicy_rollout so that we dont have to switch randomisation on and off
                        # params=env.get_dynamics_parameters()
                        # env.randomisation_off()
                        # epi rollout first for each episode

                        # TODO: Check this change -> I separated the No reset and reset further, by logging the trajectory of no reset
                        if NO_RESET:

                            i = traj_l - 1

                            if(task == 'reaching'):
                                grab_data = grab_reach_data
                            elif(task == 'pushing'):
                                grab_data = grab_push_data
                            else:
                                grab_data = None

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

                            # TODO: check this change -> only adding randomisaiton on and off when resetting the environment
                            params = env.get_dynamics_parameters()
                            env.randomisation_off()
                            env.set_dynamics_parameters(params)  # same as the rollout env
                            obs = env.reset()
                            env.randomisation_on()

                        # TODO: make sure this is corect -> for UPOSI params are concatenated before obs, here its the other way arround

                    obs = np.concatenate((obs, embedding))

                    while (True and not env.done):

                        mujoco_elapsed = env.sim.data.time - mujoco_start_time

                        #### CHOOSING THE ACTION #####
                        action = policy.get_action(obs, noise_scale=0.0)
                        ##############################

                        try:
                            next_obs, reward, done, info = env.step(action)
                        except MujocoException():
                            print ('Mujoco exceptiop')


                        if(task == 'reaching'):
                            # Grab logging data
                            eef_pos_in_base, eef_vel_in_base, goal_pos_in_base = grab_reach_data(info, world_pose_in_base)
                            action_in_base = base_rot_in_eef.dot(action)
                            logger.log(i, mujoco_elapsed,
                                       action_in_base[0], action_in_base[1], action_in_base[2],
                                       eef_pos_in_base[0], eef_pos_in_base[1], eef_pos_in_base[2],
                                       eef_vel_in_base[0], eef_vel_in_base[1], eef_vel_in_base[2],
                                       goal_pos_in_base[0], goal_pos_in_base[1], goal_pos_in_base[2],
                                       obs[0], obs[1], obs[2],
                                       obs[3], obs[4], obs[5],
                                       )
                        elif (task == 'pushing'):
                            goal_pos_in_base, eef_pos_in_base, eef_vel_in_base, \
                            object_pos_in_base, object_vel_in_base, z_angle, = grab_push_data(info, world_pose_in_base)

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
                        elif (task == 'sliding'):
                            logger.log(i, mujoco_elapsed,
                                       action[0], action[1],
                                       obs[0], obs[1], obs[2],
                                       obs[3], obs[4], obs[5], obs[6],
                                       obs[7], obs[8], obs[9],
                                       obs[10], obs[11],
                                       )
                        next_obs=np.concatenate((next_obs, embedding))

                        obs = next_obs

                        if(render):
                            env.render()

                        i += 1

                        if (i >= horizon):
                            break

    env.close()



if __name__ == "__main__":
    main()
