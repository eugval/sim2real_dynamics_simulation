import numpy as np
from robosuite_extra.env_base import make
import robosuite_extra.utils.transform_utils as T
from robosuite_extra.wrappers import EEFXVelocityControl
from sim2real_calibration_characterisation.utils.logger import Logger
from sim2real_policies.utils.choose_env import reach_full_randomisation,reach_force_noise_randomisation, \
                                                reach_force_randomisation,reach_no_randomisation


def grab_data(info, world_pose_in_base):
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


def main():
    ### Script parameters ###
    log_dir = './reach'
    render = False
    ###

    for repeat in range(50):
        # Create the environment
        env = make(
            'SawyerReach',
            gripper_type="PushingGripper",
            parameters_to_randomise=reach_no_randomisation,
            randomise_initial_conditions=False,
            table_full_size=(0.8, 1.6, 0.719),
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10,
            horizon=40,
            ignore_done=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            pid=True,
            success_radius=0.01
        )

        env = EEFXVelocityControl(env, dof=3, normalised_actions=True)

        if (render):
            env.renderer_on()

        obs = env.reset()
        print(env.get_dynamics_parameters())

        if(render):
            env.viewer.set_camera(camera_id=0)
            env.render()

        # Setup the logger
        # Base frame rotation in the end effector to convert data between those two frames
        base_rot_in_eef = env.init_right_hand_orn.T

        base_pos_in_world = env.sim.data.get_body_xpos("base")
        base_rot_in_world = env.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)



        log_list = ["step", "time",
                    "cmd_eef_vx", "cmd_eef_vy", "cmd_eef_vz",
                    "eef_x", "eef_y", "eef_z",
                    "eef_vx", "eef_vy", "eef_vz",
                    "goal_x", "goal_y", "goal_z",
                    "obs_0", "obs_1", "obs_2",
                    "obs_3", "obs_4", "obs_5"
                    ]

        log_path = '{}/trajectory{}.csv'.format(log_dir,repeat)
        logger = Logger(log_list, log_path, verbatim=render)

        i = 0
        mujoco_start_time = env.sim.data.time
        # Run the trajectories
        while (True):
            mujoco_elapsed = env.sim.data.time - mujoco_start_time

            action = np.array([0.5, 0.5, 0.5])  # policy.cos_wave(mujoco_elapsed),0.0])

            next_obs, reward, done, _ = env.step(action)

            eef_pos_in_base, eef_vel_in_base, goal_pos_in_base = grab_data(obs, world_pose_in_base)

            action_in_base = base_rot_in_eef.dot(action)

            logger.log(i, mujoco_elapsed,
                       action_in_base[0], action_in_base[1], action_in_base[2],
                       eef_pos_in_base[0], eef_pos_in_base[1], eef_pos_in_base[2],
                       eef_vel_in_base[0], eef_vel_in_base[1], eef_vel_in_base[2],
                       goal_pos_in_base[0], goal_pos_in_base[1], goal_pos_in_base[2],
                       obs['task-state'][0], obs['task-state'][1], obs['task-state'][2],
                       obs['task-state'][3], obs['task-state'][4], obs['task-state'][5],
                       )

            obs = next_obs

            if(render):
                env.render()
            i += 1
            if (mujoco_elapsed > 2.5):
                env.close()
                break


if __name__ == "__main__":
    main()
