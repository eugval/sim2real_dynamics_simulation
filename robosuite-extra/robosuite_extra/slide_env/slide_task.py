from robosuite.models.tasks import Task
from robosuite.utils.mjcf_utils import new_joint, array_to_string
import numpy as np

class SlideTask(Task):
    """
    Creates MJCF model of a slide t task.
    """

    def __init__(self, mujoco_arena, mujoco_robot,  slide_object):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            slide_object: MJCF model of the object to be slided
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(slide_object)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.merge(mujoco_arena)

    def merge_objects(self, slide_object):
        """Adds a physical object to the MJCF model."""
        self.push_object_idx = 0
        self.max_horizontal_radius = 0

        self.slide_object = slide_object


        self.merge_asset(self.slide_object)

        # Add object to xml
        obj = self.slide_object.get_collision(name="slide_object", site=True)
        obj.append(new_joint(name="slide_object", type="free"))
        self.xml_object = obj
        self.worldbody.append(obj)
        self.max_horizontal_radius = self.slide_object.get_horizontal_radius()