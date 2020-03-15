import numpy as np
from robosuite_extra.env_base import  make
from robosuite_extra.reach_env.sawyer_reach import SawyerReach
import time
from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper


goal1 = np.array([0.,0.])
goal2 = np.array([-4.465e-2,5.85e-2])
goal3 = np.array([8.37e-2,-5.78e-2])

if __name__ == "__main__":
    env=make(
        'SawyerReach',
        gripper_type="PushingGripper",
        parameters_to_randomise = [],
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

    env = FlattenWrapper( GymWrapper(EEFXVelocityControl(env, dof=3,)))
    env.renderer_on()
    env._set_goal_neutral_offset(*goal2)
    env.reset()

    if (env.has_renderer):
        env.viewer.set_camera(camera_id=0)
        env.render()

    start_time = time.time()
    for i in range(1,120):

        action = np.zeros((3,))*0.5
        action[1]=1.0

        obs, reward, done, _ = env.step(action)

        if(env.has_renderer):
            env.render()

        if(done):
            time.sleep(2)
            env.reset()
            if (env.has_renderer):
                env.viewer.set_camera(camera_id=0)
                env.render()

