"""
A version of TwoFingerGripper but always closed.
"""
import numpy as np
from robosuite.models.grippers.two_finger_gripper import TwoFingerGripper


class PushingGripper(TwoFingerGripper):
    """
    Same as TwoFingerGripper, but always closed
    """
    @property
    def init_qpos(self):
        # Fingers always closed
        return np.array([-0.020833, 0.020833])

    def format_action(self, action):
        return np.array([1, -1])

    @property
    def dof(self):
        return 0
