import numpy as np
from robosuite_extra.env_base import make
from robosuite_extra.push_env.sawyer_push import SawyerPush
import time
from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper


start_pos = np.array([0.,-16.75e-2])
goal1 = np.array([3e-3,15.15e-2])
goal2 = np.array([9.5e-2,11.01e-2])
goal3 = np.array([-11.5e-2,17.2e-2])

if __name__ == "__main__":
    env=make(
        'SawyerPush',
        gripper_type="PushingGripper",
        parameters_to_randomise = [],
        randomise_initial_conditions = False,
        table_full_size=(0.8, 1.6, 0.719),
        table_friction=(0.001, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        placement_initializer=None,
        gripper_visualization=True,
        use_indicator_object=False,
        has_renderer=True,
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
        pid = True,
    )

    env = FlattenWrapper( GymWrapper(EEFXVelocityControl(env, dof=2)))

    env._set_gripper_neutral_offset(*start_pos)
    env._set_goal_neutral_offset(*goal1)
    env.reset()

    env.viewer.set_camera(camera_id=0)
    env.render()

    start_time = time.time()
    for i in range(0,120):
        action = np.zeros(2)
        action[0]=1.0

        obs, reward, done, _ = env.step(action)
        env.render()

        if(done):
                reset_time1 = time.time()
                env.reset()
                env.viewer.set_camera(camera_id=0)
                env.render()