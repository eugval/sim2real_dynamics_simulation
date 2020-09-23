import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import array_to_string
import os

class Sawyer(Robot):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(self, torque=False):
        self.torque = torque
        file_path = os.path.dirname(os.path.realpath(__file__))
        if (torque):
            super().__init__(os.path.join(os.path.join(file_path , "../../assets/robot/sawyer/robot_torque.xml")))
        else:
            super().__init__(os.path.join(os.path.join(file_path , "../../assets/robot/sawyer/robot.xml")))

        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def links(self):
        return ["right_l{}".format(x) for x in range(7)]

    @property
    def actuators(self):
        if (self.torque):
            return ["torq_right_j{}".format(x) for x in range(7)]
        else:
            return ["vel_right_j{}".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])

    @property
    def joint_velocity_limits(self):
        return np.array([[-1.74, 1.74],
                         [-1.328, 1.328],
                         [-1.957, 1.957],
                         [-1.957, 1.957],
                         [-3.485, 3.485],
                         [-3.485, 3.485],
                         [-4.545, 4.545]])

    @property
    def velocity_pid_gains(self):
        return {'right_j0': {'p': 8.55378248e+01, 'i': 4.58211002e+00, 'd':  9.80858595e-02},
                    'right_j1': {'p': 7.72400386e+01, 'i': 2.57669899e+00, 'd':  4.23858218e-02},
                    'right_j2': {'p':5.35079700e+01, 'i':4.30959305e+00, 'd': 0.00000000e+00},
                    'right_j3': {'p': 4.88937010e+01, 'i':2.91438891e+00, 'd': 4.52197212e-02},
                    'right_j4': {'p': 3.57707442e+01, 'i': 3.14960177e+00, 'd': 2.03120628e-01},
                    'right_j5': {'p':  2.82424289e+01, 'i':2.18658812e+00, 'd': 2.03629986e-01},
                    'right_j6': {'p': 1.62834673e+01, 'i': 1.71284099e+00, 'd':  8.44274724e-02},
                    }


    @property
    def position_pid_gains(self):

        # array([77.39238042, 85.93730043, 53.35490038, 57.10317523, 27.80825017,
        # 3.17180638,  2.90868948,  4.57497262,  3.56435536,  2.47908628,
        # 0.45245024,  0.31105989,  0.83930337,  0.76906677,  0.42219411])

        return {'right_j0': {'p': 77.39238042, 'i': 3.17180638, 'd':  0.45245024},
                    'right_j1': {'p':  85.93730043, 'i': 2.90868948, 'd':   0.31105989},
                    'right_j2': {'p': 53.35490038, 'i':4.57497262, 'd':  0.83930337},
                    'right_j3': {'p':  57.10317523, 'i':3.56435536, 'd': 0.76906677},
                    'right_j4': {'p':  27.808250171, 'i':  2.47908628, 'd':  0.42219411},
                    'right_j5': {'p':  2.82424289e+01, 'i':2.18658812e+00, 'd': 2.03629986e-01},
                    'right_j6': {'p': 1.62834673e+01, 'i': 1.71284099e+00, 'd':  8.44274724e-02},
                    }

        # return {'right_j0': {'p': 8.55378248e+01, 'i': 4.58211002e+00, 'd': 9.80858595e-02},
        #         'right_j1': {'p': 7.72400386e+01, 'i': 2.57669899e+00, 'd': 4.23858218e-02},
        #         'right_j2': {'p': 5.35079700e+01, 'i': 4.30959305e+00, 'd': 0.00000000e+00},
        #         'right_j3': {'p': 4.88937010e+01, 'i': 2.91438891e+00, 'd': 4.52197212e-02},
        #         'right_j4': {'p': 3.57707442e+01, 'i': 3.14960177e+00, 'd': 2.03120628e-01},
        #         'right_j5': {'p': 2.82424289e+01, 'i': 2.18658812e+00, 'd': 2.03629986e-01},
        #         'right_j6': {'p': 1.62834673e+01, 'i': 1.71284099e+00, 'd': 8.44274724e-02},
        #     }


