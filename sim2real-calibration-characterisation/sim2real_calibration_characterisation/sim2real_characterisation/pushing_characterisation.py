import numpy as np
from robosuite_extra.env_base import make
import robosuite_extra.utils.transform_utils as T
from robosuite_extra.wrappers import EEFXVelocityControl
from sim2real_calibration_characterisation.utils.logger import Logger
from sim2real_policies.utils.choose_env import push_force_noise_randomisation, push_force_randomisation, \
                                           push_full_randomisation,push_no_randomisation

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
    ### Script parameters ###
    render = False
    log_dir = './push'
    ####


    action_ref = np.array([0.1, 0.0])
    action = action_ref

    for repeat in range(10):
        # Create the environment
        env = make(
            'SawyerPush',
            gripper_type="PushingGripper",
            parameters_to_randomise=push_no_randomisation,
            randomise_initial_conditions=False,
            table_full_size=(0.8, 1.6, 0.719),
            table_friction=(0.001, 5e-3, 1e-4),
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            placement_initializer=None,
            gripper_visualization=True,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10.,
            horizon=80,
            ignore_done=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
        )

        env = EEFXVelocityControl(env, dof=2, normalised_actions=False)

        if (render):
            env.renderer_on()

        obs = env.reset()
        if (render):
            env.viewer.set_camera(camera_id=0)
            env.render()

        ## Setup the logger
        # Base frame rotation in the end effector to convert data between those two frames
        base_rot_in_eef = env.init_right_hand_orn.T

        base_pos_in_world = env.sim.data.get_body_xpos("base")
        base_rot_in_world = env.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        world_pose_in_base = T.pose_inv(world_pose_in_base)
        log_list = ["step", "time",
                    "cmd_eef_vx", "cmd_eef_vy",
                    "goal_x", "goal_y", "goal_z",
                    "eef_x", "eef_y", "eef_z",
                    "eef_vx", "eef_vy", "eef_vz",
                    "object_x", "object_y", "object_z",
                    "object_vx", "object_vy", "object_vz",
                    "z_angle", "obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "obs_5", "obs_6", "obs_7", "obs_8", "obs_9"  ]

        log_path = '{}/trajectory_{}.csv'.format(log_dir, repeat)

        logger = Logger(log_list, log_path,verbatim=render)

        ## Run the trajectory
        i = 0
        mujoco_start_time = env.sim.data.time
        while (True):
            mujoco_elapsed = env.sim.data.time - mujoco_start_time

            next_obs, reward, done, _ = env.step(action)

            goal_pos_in_base, eef_pos_in_base, eef_vel_in_base, \
            object_pos_in_base, object_vel_in_base, z_angle, = grab_data(obs, world_pose_in_base)

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
                       obs['task-state'][0], obs['task-state'][1], obs['task-state'][2],
                       obs['task-state'][3], obs['task-state'][4], obs['task-state'][5],
                       obs['task-state'][6],  obs['task-state'][7],  obs['task-state'][8], obs['task-state'][9],
                       )
            obs = next_obs

            if(render):
                env.render()
            i += 1
            if (mujoco_elapsed >2.5):
                action =np.array([0.0, 0.0])

            if(mujoco_elapsed >3.0):
                action = action_ref
                env.close()
                break


if __name__ == "__main__":
    main()
