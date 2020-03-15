import numpy as np
from robosuite_extra.env_base import make
from sim2real_calibration_characterisation.utils.logger import Logger
from sim2real_policies.utils.choose_env import  slide_force_randomisation, slide_force_noise_randomisation,\
                                                 slide_full_randomisation,slide_no_randomisation

def main():
    ### Script parameters ###
    render = False
    log_dir = './slide'
    ####

    action_ref = np.array([0.1, 0.1])
    action = action_ref

    for repeat in range(10):
        # Create the environment
        env = make(
            'SawyerSlide',
            gripper_type="SlidePanelGripper",
            parameters_to_randomise=slide_no_randomisation,
            randomise_initial_conditions=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            use_indicator_object=False,
            has_renderer=False,
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

        env.normalised_actions = False

        if (render):
            env.renderer_on()

        obs = env.reset()

        if (render):
            env.viewer.set_camera(camera_id=0)
            env.render()

        # Setup the logger
        log_list =["step", "time",
                    "cmd_j5", "cmd_j6",
                    "obj_x", "obj_y", "obj_z",
                     "sin_z", "cos_z",
                    "obj_vx", "obj_vy", "obj_vz",
                    "a_j5", "a_j6",
                    "v_j5", "v_j6",
                    ]
        log_path = '{}/trajectory_{}.csv'.format(log_dir, repeat)
        logger = Logger(log_list, log_path,verbatim=render)

        i = 0
        mujoco_start_time = env.sim.data.time
        # Run the trajectories
        while (True):
            mujoco_elapsed = env.sim.data.time - mujoco_start_time

            next_obs, reward, done, info = env.step(action)
            assert len( obs['task-state']) ==12

            logger.log(i, mujoco_elapsed,
                       action[0], action[1],
                        obs['task-state'][0],  obs['task-state'][1],  obs['task-state'][2],
                       obs['task-state'][3],  obs['task-state'][4],  obs['task-state'][5],  obs['task-state'][6],
                       obs['task-state'][7],  obs['task-state'][8],  obs['task-state'][9],
                       obs['task-state'][10],  obs['task-state'][11],
                       )
            obs = next_obs

            if(render):
                env.render()
            i += 1
            if (mujoco_elapsed >2.):
                break


if __name__ == "__main__":
    main()
