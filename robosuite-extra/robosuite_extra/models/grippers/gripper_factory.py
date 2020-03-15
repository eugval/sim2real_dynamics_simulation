"""
Overrides the girpper factory of robosuite in order to add the new grippers for our tasks
"""

from robosuite.models.grippers  import gripper_factory as robosuite_gripper_factory
from .pushing_gripper import PushingGripper
from .slide_panel_gripper import SlidePanelGripper

def gripper_factory(name):
    if name == "PushingGripper":
        # Closed two finger gripper
        return PushingGripper()
    if name == "SlidePanelGripper":
        return SlidePanelGripper()
    else:
        return robosuite_gripper_factory(name)



