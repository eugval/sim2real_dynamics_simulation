"""

Taken and  modified from the original robosuite repository (version 0.1.0)
Our fork with version 0.1.0 : https://github.com/eugval/robosuite
Official Robosuite Repository : https://github.com/ARISE-Initiative/robosuite




This module implements a wrapper for controlling the robot through end effector
movements instead of joint velocities. This is useful in learning pipelines
that want to output actions in end effector space instead of joint space.
"""

import numpy as np
from robosuite.wrappers import Wrapper
from robosuite_extra.controllers import SawyerEEFVelocityController


class EEFXVelocityControl(Wrapper):
    env = None

    def __init__(self, env, dof = 2, max_action = 0.1, normalised_actions = True):
        """
        End effector cartesian velocity control wrapper.

        This wrapper allows for controlling the robot by sending end-effector cartesian velocity commands
        instead of joint velocities. This includes velocity commands in-plance (v_x,v_y) or in 3D (v_x,v_y,v_z).
        Does not implement angular velocity commands.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            dof (int 2, or 3): Whether to move the end-effector in-plane or in 3d
            max_action : The maximum speed that can be achieved in each dimension (in m/sec).

        """
        super().__init__(env)
        self.controller = SawyerEEFVelocityController()
        assert dof == 2 or dof == 3 , "Can only send x-y or x-y-z velocity commands"

        self.wrapper_dof = dof
        self.action_range = [-max_action, max_action]

        self.normalised_actions = normalised_actions


        #Desable the action spec
        self.env.normalised_actions = False


    @property
    def action_spec(self):
        if(self.normalised_actions):
            low = np.ones(self.wrapper_dof) * -1
            high = np.ones(self.wrapper_dof)
        else:
            low = np.array([self.action_range[0]]*self.wrapper_dof )
            high = np.array([self.action_range[1]]*self.wrapper_dof )
        return low, high

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)


    def step(self, action):
        if self.env.done:
            raise ValueError("Executing action in terminated episode")

        low, high = self.action_spec

        # Domain randomisation for the action space
        action_range = high-low
        additive_noise = action_range*self.env.dynamics_parameters['action_additive_noise'] * self.env.np_random.uniform(-1, 1, action.shape)
        additive_systematic_noise = action_range*self.dynamics_parameters['action_systematic_noise']
        multiplicative_noise = 1.0 + (
                    self.dynamics_parameters['action_multiplicative_noise'] * self.env.np_random.uniform(-1, 1,
                                                                                                     action.shape))


        action = action*multiplicative_noise +additive_noise+additive_systematic_noise

        action = np.clip(action, low, high)

        if (self.normalised_actions):
            # Rescale action to -max_action, max_action

            # rescale normalized action to control ranges
            bias = 0.5 * (self.action_range[1] + self.action_range[0])
            weight = 0.5 * (self.action_range[1] - self.action_range[0])
            action = bias + weight * action

        current_right_hand_pos_base = self.env._right_hand_pos
        current_right_hand_pos_eef = self.env.init_right_hand_orn.dot(current_right_hand_pos_base)


        if(self.wrapper_dof == 2):
            #This only works for the pushing task
            # Correct for deviation in the z-position
            # Compensate for the fact that right_hand_pos is in the base frame by a minus sign (current - initial)

            # action = self.compensate_for_external_forces(action, current_right_hand_pos_eef)
            z_vel =   current_right_hand_pos_base[2] - self.env.init_right_hand_pos[2]
            xyz_vel = np.concatenate([action, [2*z_vel]])

        elif(self.wrapper_dof ==3):
            xyz_vel = action
        else:
            raise NotImplementedError('Dof can only be 2 or 3!')

        self.target_next_state = current_right_hand_pos_eef + xyz_vel* self.env.control_timestep

        current_right_hand_orn = self.env._right_hand_orn
        reference_right_hand_orn = self.env.init_right_hand_orn

        orn_diff = reference_right_hand_orn.dot(current_right_hand_orn.T)
        orn_diff_twice = orn_diff.dot(orn_diff).dot(orn_diff)



        # Convert xyz_vel  from the end-effector frame to the base_frame of the robot
        base_frame_in_eef = reference_right_hand_orn.T
        xyz_vel_base = base_frame_in_eef.dot(xyz_vel)


        pose_matrix = np.zeros((4,4))
        pose_matrix[:3,:3] = orn_diff_twice
        pose_matrix[:3,3] = xyz_vel_base
        pose_matrix[3,3] =1
        current_joint_angles = self._robot_jpos_getter()



        joint_velocities = self.controller.compute_joint_velocities_for_endpoint_velocity(pose_matrix,current_joint_angles)
        final_action = np.concatenate([np.asarray(joint_velocities).squeeze()])

        obs, reward, done, info =  self.env.step(final_action)
        info['joint_velocities'] = final_action
        return obs,reward,done,info


    def reset(self):
        ret = self.env.reset()
        eff_frame_in_base = self.env.init_right_hand_orn
        self.target_next_state = eff_frame_in_base.dot(self.env._right_hand_pos)
        return ret




