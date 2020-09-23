import numpy as np
from robosuite_extra.env_base import make
import robosuite_extra.utils.transform_utils as T
from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper
from sim2real_policies.final_policy_testing.network_loading import load, load_model
from sim2real_policies.utils.logger import Logger
import os
from sim2real_policies.utils.choose_env import reach_full_randomisation, push_full_randomisation, slide_full_randomisation
from sim2real_policies.sys_id.common.utils import query_params
from sim2real_policies.sys_id.universal_policy_online_system_identification.osi import stack_data


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

    return env


def main():
    render = False

    # Parameters
    total_repeats = 50


    for env_name in ['SawyerSlide', 'SawyerReach','SawyerPush' ]: #, 'SawyerReach', 'SawyerSlide'
        for use_white_noise in [ False]: # True
            if(not use_white_noise):
                parameter_predictions = []
                oracle_parameters = []

            if(env_name == 'SawyerReach'):
                state_dim = 6
                action_dim = 3
                horizon = 50
                task = 'reaching'
            elif(env_name == 'SawyerPush'):
                horizon = 80
                state_dim = 10
                action_dim = 2
                task = 'pushing'
            elif(env_name == 'SawyerSlide'):
                horizon = 60
                state_dim = 12
                action_dim = 2
                task = 'sliding'

            env = make_env(task,render)
            env.reset()

            osi_l = 5

            method = 'UPOSI'

            RANDOMISZED_ONLY = True
            DYNAMICS_ONLY = True
            CAT_INTERNAL = True
            if (task == 'sliding'):
                CAT_INTERNAL = False

            params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
            params_dim = params.shape[0]  # dimension of parameters for prediction
            if CAT_INTERNAL:
                internal_state_dim = env.get_internal_state_dimension()
                _, _, _, info = env.step(np.zeros(action_dim))
                internal_action_dim = np.array(info["joint_velocities"]).shape[0]
                osi_input_dim = osi_l * (state_dim + action_dim + internal_state_dim + internal_action_dim)
            else:
                osi_input_dim = osi_l * (state_dim + action_dim)

            state_dim += params_dim

            alg_name = 'uposi_td3'

            if(task == 'reaching'):
                GOALS = REACH_GOALS
            elif(task == 'pushing'):
                GOALS = PUSH_GOALS
            elif(task == 'sliding'):
                GOALS = SLIDE_GOALS

            for goal_idx, goal_pos in enumerate(GOALS):
                goal_idx = goal_idx+1
                ###### SETTING UP THE ENVIRONMENT ######
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

                # choose which randomisation is applied
                randomisation_type = 'reach_full-randomisation'
                if (task == 'reaching'):
                    number_random_params = 14
                elif (task == 'pushing'):
                    number_random_params = 23
                elif (task == 'sliding'):
                    number_random_params = 22

                path = '../../../../sawyer/src/sim2real_dynamics_sawyer/assets/rl/'+method +'/' + alg_name + '/model/' + env_name + str(
                    number_random_params) + '_' + alg_name
                try:
                    policy = load(path=path, alg=alg_name, state_dim=state_dim,
                                  action_dim=action_dim)

                    if(not use_white_noise):
                        osi_model = load_model(model_name='osi', path=path, input_dim=osi_input_dim,
                                               output_dim=params_dim)
                except :
                    print(method,',',randomisation_type)

                    continue

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

                    log_path = '../../../../data/uposi_ablation/{}/{}/{}/goal_{}/trajectory_log_{}.csv'.format(task, noise_folder,  log_save_name,goal_idx,repeat)

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

                    if(use_white_noise):
                        params = np.random.uniform(-1.,1.,size =(params_dim,))
                        params_state = np.concatenate((params, obs))
                    else:
                        epi_traj = []
                        params = query_params(env, randomised_only=RANDOMISZED_ONLY, dynamics_only=DYNAMICS_ONLY)
                        zero_osi_input = np.zeros(osi_input_dim)
                        pre_params = osi_model(zero_osi_input).detach().numpy()

                        oracle_parameters.append(params)
                        parameter_predictions.append(pre_params)

                        params_state = np.concatenate((pre_params, obs))


                    while (True):

                        mujoco_elapsed = env.sim.data.time - mujoco_start_time

                        #### CHOOSING THE ACTION #####

                        if CAT_INTERNAL:
                            internal_state = env.get_internal_state()
                            full_state = np.concatenate([obs, internal_state])
                        else:
                            full_state = obs

                        action = policy.get_action(params_state, noise_scale=0.0)

                        ##############################

                        try:
                            next_obs, reward, done, info = env.step(action)
                        except MujocoException():
                            print ('Mujoco exceptiop')

                        if (use_white_noise):
                            params = np.random.uniform(-1., 1., size = (params_dim,))
                            next_params_state = np.concatenate((params, next_obs))
                            params_state = next_params_state
                        else:
                            if CAT_INTERNAL:
                                target_joint_action = info["joint_velocities"]
                                full_action = np.concatenate([action, target_joint_action])
                            else:
                                full_action = action
                            epi_traj.append(np.concatenate((full_state, full_action)))

                            if len(epi_traj)>=osi_l:
                                osi_input = stack_data(epi_traj, osi_l)
                                pre_params = osi_model(osi_input).detach().numpy()
                            else:
                                zero_osi_input = np.zeros(osi_input_dim)
                                pre_params = osi_model(zero_osi_input).detach().numpy()

                            oracle_parameters.append(params)
                            parameter_predictions.append(pre_params)

                            next_params_state = np.concatenate((pre_params, next_obs))
                            params_state = next_params_state



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

                        obs = next_obs

                        if(render):
                            env.render()

                        i += 1
                        if (i >= horizon):
                            break


            if(not use_white_noise):
                parameter_predictions = np.array(parameter_predictions)
                oracle_parameters = np.array(oracle_parameters)
                percent_diff = np.abs((parameter_predictions-oracle_parameters)/2.)*100

                average_percent_diff = np.nanmean(percent_diff[np.isfinite(percent_diff)])
                print('{} percent diff :{}'.format(task,average_percent_diff))
                save_path = '../../../../data/uposi_ablation/{}/{}/'.format(task,noise_folder)
                if(os.path.exists(save_path)):
                    file = open(os.path.join(save_path, 'percent_diff.txt'), 'w')
                else:
                    file = open(os.path.join(save_path, 'percent_diff.txt'), 'a')
                file.write('{} percent diff :{}'.format(task,average_percent_diff))
                file.close()

    env.close()



if __name__ == "__main__":
    main()
