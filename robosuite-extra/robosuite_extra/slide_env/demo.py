import numpy as np
from robosuite_extra.env_base import make
from robosuite_extra.slide_env.sawyer_slide import SawyerSlide
import time

if __name__ == "__main__":
    env=make(
        'SawyerSlide',
        gripper_type="SlidePanelGripper",
        parameters_to_randomise = [],
        randomise_initial_conditions = True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        use_indicator_object=False,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=30,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        pid=True,
    )

    env.reset()
    env.viewer.set_camera(camera_id=0)

    env.render()
    for i in range(1,20000):

        action = np.zeros(2)
        action[0]  = 1.
        obs, reward, done, _ = env.step(action)
        env.render()

        if(done):
                time.sleep(2)
                env.reset()
                env.viewer.set_camera(camera_id=0)
                env.render()
