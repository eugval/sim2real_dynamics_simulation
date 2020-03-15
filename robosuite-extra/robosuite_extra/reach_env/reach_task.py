from robosuite.models.tasks import Task
class ReachTask(Task):
    """
    Creates MJCF model of a push task.

    A tabletop task consists of one robot pushing an object to a visual goal on a
     tabletop. This class combines the robot, the table
    arena, and the objetcts into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_goal):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_goal: MJCF model of the goal
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_visual(mujoco_goal)


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

    def merge_visual(self, mujoco_goal):
        """Adds the visual goal to the MJCF model."""
        self.mujoco_goal = mujoco_goal
        self.merge_asset(mujoco_goal)
        # Load goal
        goal_visual = mujoco_goal.get_visual(name='goal', site=True)

        self.xml_goal = goal_visual

        self.worldbody.append(goal_visual)


