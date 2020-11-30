'''
Taken and  modified from the original robosuite repository (version 0.1.0)
Our fork with version 0.1.0 : https://github.com/eugval/robosuite
Official Robosuite Repository : https://github.com/ARISE-Initiative/robosuite

'''

import numpy as np

import robosuite_extra.utils.transform_utils as T
from robosuite_extra.env_base import MujocoEnv

from robosuite_extra.models.grippers import gripper_factory
from robosuite_extra.models.robot import Sawyer
from simple_pid import PID
import copy

class SawyerEnv(MujocoEnv):
    """Initializes a Sawyer robot environment."""

    def __init__(
            self,
            gripper_type=None,
            gripper_visualization=False,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10,
            horizon=1000,
            ignore_done=False,
            use_camera_obs=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            pid=True
    ):
        """
        Args:
            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        self.has_gripper = gripper_type is not None
        self.gripper_type = gripper_type
        self.gripper_dof = 0
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object


        # Set to False if actions are supposed to be applied as is
        self.normalised_actions = True

        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

        if (pid):
            self._setup_pid()


    def _setup_pid(self):
        self.pid = PID(dim= self.mujoco_robot.dof, sample_time=None, )
        limits = (self.sim.model.actuator_ctrlrange[:7, 0].copy(), self.sim.model.actuator_ctrlrange[:7, 1].copy())
        gains = self.mujoco_robot.velocity_pid_gains
        kps = np.array([gains['right_j{}'.format(actuator)]['p'] for actuator in range(7)])
        kis = np.array([gains['right_j{}'.format(actuator)]['i'] for actuator in range(7)])
        kds = np.array([gains['right_j{}'.format(actuator)]['d'] for actuator in range(7)])

        self.pid.tunings = (kps, kis, kds)
        self.pid.output_limits = limits


    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        if (self.pid is not None):
            self.mujoco_robot = Sawyer(torque=True)
        else:
            self.mujoco_robot = Sawyer()

        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)
            self.gripper_dof = self.gripper.dof

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos

        if self.has_gripper:
            self.sim.data.qpos[
                self._ref_gripper_joint_pos_indexes
            ] = self.gripper.init_qpos

    def _get_reference(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_ids = [self.sim.model.joint_name2id(x) for x in self.robot_joints]

        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]

            self._ref_gripper_body_indx = self.sim.model.body_name2id('right_gripper')

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        # self._ref_joint_pos_actuator_indexes = [
        #     self.sim.model.actuator_name2id(actuator)
        #     for actuator in self.sim.model.actuator_names
        #     if actuator.startswith("pos")
        # ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        self._ref_joint_torque_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("torq")
        ]

        if self.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper")
            ]

        # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id("grip_site")
        self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def move_indicator(self, pos):
        """
        Sets 3d position of indicator object to @pos.
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low
            self.sim.data.qpos[index: index + 3] = pos


    def _pre_action(self, action):
            """
            Overrides the superclass method to actuate the robot with the
            passed joint velocities and gripper control.

            Args:
                action (numpy array): The control to apply to the robot. The first
                    @self.mujoco_robot.dof dimensions should be the desired
                     joint velocities. If self.normalised_actions is True then these actions should be normalised.
                     If the robot has a gripper, the next @self.gripper.dof dimensions should be
                    actuation controls for the gripper, which should always be normalised.
            """
            assert len(action) == self.dof, "environment got invalid action dimension"
            low, high = self.action_spec


            arm_action = action[: self.mujoco_robot.dof]
            gripper_action = action[
                                self.mujoco_robot.dof: self.mujoco_robot.dof + self.gripper_dof
                                ]

            if(self.normalised_actions):
                arm_action = np.clip(arm_action, low[:self.mujoco_robot.dof], high[:self.mujoco_robot.dof])

                # rescale normalized action to control ranges
                ctrl_range = self.control_range[:self.mujoco_robot.dof, :]
                bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
                weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
                arm_action = bias + weight * arm_action


            if self.has_gripper:
                gripper_action = self.gripper.format_action(gripper_action)
                # rescale normalized action to control ranges
                ctrl_range = self.control_range[self.mujoco_robot.dof:, :]
                bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
                weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
                gripper_action = bias + weight * gripper_action

                action = np.concatenate([arm_action, gripper_action])
            else:
                action = arm_action


            if(self.pid is None):
                self.sim.data.ctrl[:] = action
            else:
                self.sim.data.ctrl[self.mujoco_robot.dof:] = gripper_action
                self.pid.reset()
                self.pid.setpoint = arm_action


            # gravity compensation
            self.sim.data.qfrc_applied[
                self._ref_joint_vel_indexes
            ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]


            if self.use_indicator_object:
                self.sim.data.qfrc_applied[
                self._ref_indicator_vel_low: self._ref_indicator_vel_high
                ] = self.sim.data.qfrc_bias[
                    self._ref_indicator_vel_low: self._ref_indicator_vel_high
                    ]

    def _set_pid_control(self):
        dt = self.model_timestep if self.pid._last_output is not None else 1e-16

        current_qvel = self.sim.data.qvel[self._ref_joint_vel_indexes]
        self.sim.data.ctrl[:self.mujoco_robot.dof] = self.pid(current_qvel, dt)

    def _post_action(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
        ret = super()._post_action(action)
        self._gripper_visualization()
        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """

        di = super()._get_observation()
        # proprioceptive features
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        robot_states = [
            np.sin(di["joint_pos"]),
            np.cos(di["joint_pos"]),
            di["joint_vel"],
        ]

        if self.has_gripper:
            di["gripper_qpos"] = np.array(
                [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
            )
            di["gripper_qvel"] = np.array(
                [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
            )

            di["eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )

            # add in gripper information
            if (self.gripper_type == "PushingGripper"):
                robot_states.extend([di["eef_pos"], di["eef_quat"]])
            else:
                robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"]])

        di["robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def control_range(self):
        return self.sim.model.actuator_ctrlrange

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        if self.normalised_actions:
            low = np.ones(self.dof) * -1.
            high = np.ones(self.dof) * 1.
            return low, high
        else:
            ctrl_range = self.control_range
            return ctrl_range[:,0], ctrl_range[:,1]

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        dof = self.mujoco_robot.dof
        if self.has_gripper:
            dof += self.gripper.dof
        return dof

    def pose_in_base(self, pose_in_world):
        """
        A helper function that takes in a pose in world frame and returns that pose in  the
        the base frame.
        """
        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame
        as a numpy array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return copy.deepcopy(self.sim.data.qpos[self._ref_joint_pos_indexes])

    @property
    def joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return copy.deepcopy(self.sim.data.qvel[self._ref_joint_vel_indexes])

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def _joint_velocities_dict(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return {'right_j0': self.sim.data.get_joint_qvel('right_j0'),
                'right_j1': self.sim.data.get_joint_qvel('right_j1'),
                'right_j2': self.sim.data.get_joint_qvel('right_j2'),
                'right_j3': self.sim.data.get_joint_qvel('right_j3'),
                'right_j4': self.sim.data.get_joint_qvel('right_j4'),
                'right_j5': self.sim.data.get_joint_qvel('right_j5'),
                'right_j6': self.sim.data.get_joint_qvel('right_j6'),
                }

    @property
    def _joint_pos_dict(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return {'right_j0': self.sim.data.get_joint_qpos('right_j0'),
                'right_j1': self.sim.data.get_joint_qpos('right_j1'),
                'right_j2': self.sim.data.get_joint_qpos('right_j2'),
                'right_j3': self.sim.data.get_joint_qpos('right_j3'),
                'right_j4': self.sim.data.get_joint_qpos('right_j4'),
                'right_j5': self.sim.data.get_joint_qpos('right_j5'),
                'right_j6': self.sim.data.get_joint_qpos('right_j6'),
                }

    @property
    def _joint_ranges(self):
        """
        Returns a numpy array of joint ranges.
        Sawyer robots have 7 joints with max ranges in radiants.
        """
        return self.sim.model.jnt_range[self._ref_joint_ids]

    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """

        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
