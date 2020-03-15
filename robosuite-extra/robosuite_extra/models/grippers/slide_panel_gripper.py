"""
A version of TwoFingerGripper but always closed.
"""
import numpy as np
import os
from robosuite.models.grippers.gripper import Gripper

class SlidePanelGripper(Gripper):
    """
    Same as TwoFingerGripper, but always closed
    """

    def __init__(self):
        file_path = os.path.dirname(os.path.realpath(__file__))
        super().__init__(os.path.join(os.path.join(file_path , "../../assets/grippers/slide_panel_gripper.xml")))

    def format_action(self, action):
        return np.array([])

    @property
    def init_qpos(self):
       return np.array([])

    @property
    def joints(self):
        return []

    @property
    def dof(self):
        return 0

    @property
    def visualization_sites(self):
        return ["grip_site", "slide_panel_centre", "grip_site_cylinder" ]

    def contact_geoms(self):
        return [
            "r_finger_g0",
            "r_finger_g1",
            "l_finger_g0",
            "l_finger_g1",
            "r_fingertip_g0",
            "l_fingertip_g0",
            "slide_panel_g"
        ]

    @property
    def contact_geom2body(self):
        return {
                    "r_finger_g0": "r_gripper_r_finger",
                    "r_finger_g1": "r_gripper_r_finger",
                    "l_finger_g0": "r_gripper_l_finger",
                    "l_finger_g1": "r_gripper_l_finger",
                    "r_fingertip_g0": "r_gripper_r_finger_tip",
                    "l_fingertip_g0": "r_gripper_l_finger_tip",
                    "slide_panel_g": "slide_panel"
                }

    def get_contact_body_from_geom(self, geom):
       return self.contact_geom2body[geom]

    @property
    def left_finger_geoms(self):
        return ["l_finger_g0", "l_finger_g1", "l_fingertip_g0"]

    @property
    def right_finger_geoms(self):
        return ["r_finger_g0", "r_finger_g1", "r_fingertip_g0"]


