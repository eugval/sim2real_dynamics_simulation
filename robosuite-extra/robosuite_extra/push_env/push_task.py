from robosuite.models.tasks import Task
from robosuite_extra.models.tasks import UniformSelectiveSampler
from robosuite.utils.mjcf_utils import new_joint, array_to_string
import numpy as np

class PushTask(Task):
    """
    Creates MJCF model of a push task.

    A tabletop task consists of one robot pushing an object to a visual goal on a
     tabletop. This class combines the robot, the table
    arena, and the objetcts into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_goal, mujoco_objects,  initializer):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: MJCF model of the objects to be pushed
            mujoco_goal: MJCF model of the goal
            initializer: placement sampler to initialize object positions.
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        self.merge_visual(mujoco_goal)
        if initializer is None:
            initializer = UniformSelectiveSampler()


        self.initializer = initializer
        self.initializer.setup(self.mujoco_objects, self.table_top_offset, self.table_size)
        self.change_push_object()

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects ):
        """Adds a physical object to the MJCF model."""
        self.xml_objects = []
        self.mujoco_objects = []
        self.object_names = []

        self.push_object_idx = 0
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.mujoco_objects.append(obj_mjcf)
            self.object_names.append(obj_name)

            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))
            self.xml_objects.append(obj)
            self.worldbody.append(obj)
            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )


    def merge_visual(self, mujoco_goal):
        """Adds the visual goal to the MJCF model."""
        self.mujoco_goal = mujoco_goal
        self.merge_asset(mujoco_goal)
        # Load goal
        goal_visual = mujoco_goal.get_visual(name='goal', site=True)

        self.xml_goal = goal_visual

        self.worldbody.append(goal_visual)


    def change_push_object(self, idx = None):
        if(idx is None):
            idx = np.random.choice(len(self.mujoco_objects))

        assert idx < len(self.mujoco_objects), "idx out of range for selecting pushing object"

        self.push_object_idx = idx


    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""

        pos_arr, quat_arr = self.initializer.sample(self.push_object_idx)
        for i in range(len(self.xml_objects)):
            self.xml_objects[i].set("pos", array_to_string(pos_arr[i]))
            self.xml_objects[i].set("quat", array_to_string(quat_arr[i]))

