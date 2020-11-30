'''
Taken and  modified from the original robosuite repository (version 0.1.0)
Our fork with version 0.1.0 : https://github.com/eugval/robosuite
Official Robosuite Repository : https://github.com/ARISE-Initiative/robosuite

'''

from collections import OrderedDict
import numpy as np
from robosuite_extra.env_base import SawyerEnv
from robosuite.models.arenas import EmptyArena
from robosuite_extra.models.generated_objects import FullyFrictionalBoxObject
from robosuite_extra.slide_env.slide_task import SlideTask
from robosuite_extra.utils import transform_utils as T
from robosuite_extra.controllers import SawyerEEFVelocityController
import copy
from collections import deque
import mujoco_py
from simple_pid import PID


class SawyerSlide(SawyerEnv):
    """
    This class corresponds to a Pushing task for the sawyer robot arm.

    This task consists of pushing a rectangular puck from some initial position to a final goal.
    The goal and initial positions are chosen randomly within some starting bounds
    """

    def __init__(
            self,
            gripper_type="SlidePanelGripper",
            parameters_to_randomise=None,
            randomise_initial_conditions=True,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            use_indicator_object=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=10,
            horizon=80,
            ignore_done=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            pid=True,
    ):
        """
        Args:

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            randomise_parameters (bool) : Whether to use domain randomisation

            heavy_randomisation (bool) : Whether to also randomise internal arm parameters

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

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

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            pid (bool) : True if using a velocity PID controller for controlling the arm, false if using a
            mujoco-implemented proportional controller.
        """

        self.initialised = False

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping
        if (self.reward_shaping):
            self.reward_range = [-np.inf, horizon * (0.1)]
        else:
            self.reward_range = [0, 1]

        # Domain Randomisation Parameters
        self.parameters_to_randomise = parameters_to_randomise
        self.randomise_initial_conditions = randomise_initial_conditions
        self.dynamics_parameters = OrderedDict()
        self.default_dynamics_parameters = OrderedDict()
        self.parameter_sampling_ranges = OrderedDict()
        self.factors_for_param_randomisation = OrderedDict()

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=False,
            use_indicator_object=use_indicator_object,
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
            pid=pid,
        )

        self._set_default_dynamics_parameters(pid)
        self._set_default_parameter_sampling_ranges()
        self._set_dynamics_parameters(self.default_dynamics_parameters)
        self._set_factors_for_param_randomisation(self.default_dynamics_parameters)

        # Check that the parameters to randomise are within the allowed parameters
        if (self.parameters_to_randomise is not None):
            self._check_allowed_parameters(self.parameters_to_randomise)

        # IK solver for placing the arm at desired locations during reset
        self.IK_solver = SawyerEEFVelocityController()

        self.init_control_timestep = self.control_timestep
        self.init_qpos = self.mujoco_robot.init_qpos

        self.object_random_forces = np.zeros((6,))
        self.joint_random_forces = np.zeros((2,))
        self.joint_acceleration_forces = np.zeros((2,))

        self.initialised = True
        self.reset()

    def _joint_velocities_range_dict(self):
        return {
            'right_j0': (-0.3, 0.3),
            'right_j1': (-0.3, 0.3),
            'right_j2': (-0.3, 0.3),
            'right_j3': (-0.3, 0.3),
            'right_j4': (-0.3, 0.3),
            'right_j5': (-0.3, 0.3),
            'right_j6': (-0.3, 0.3),
        }

    def _joint_velocities_range(self):
       return np.ones((7,)) * -0.3, np.ones((7,)) * 0.3

    def _set_dynamics_parameters(self, parameters):
        self.dynamics_parameters = copy.deepcopy(parameters)

    def _default_damping_params(self):
        return np.array([8.19520686e-01, 1.25425414e+00, 1.04222253e+00,
                         0.00000000e+00, 1.43146116e+00, 1.26807887e-01, 1.53680244e-01, ])

    def _default_armature_params(self):
        return np.array([0.00000000e+00, 0.00000000e+00, 2.70022664e-02, 5.35581203e-02,
                         3.31204140e-01, 2.59623415e-01, 2.81964631e-01, ])

    def _default_joint_friction_params(self):
        return np.array([4.14390483e-03,
                         9.30938506e-02, 2.68656509e-02, 0.00000000e+00, 0.00000000e+00,
                         4.24867204e-04, 8.62040317e-04])

    def _set_default_dynamics_parameters(self, use_pid):
        """
        Setting the the default environment parameters.
        """
        self.default_dynamics_parameters['joint_forces'] = np.zeros((2,))
        self.default_dynamics_parameters['acceleration_forces'] = np.zeros((2,))
        self.default_dynamics_parameters['obj_forces'] = np.zeros((6,))

        self.default_dynamics_parameters['joints_timedelay'] = np.asarray(0)
        self.default_dynamics_parameters['obj_timedelay'] = np.asarray(0)
        self.default_dynamics_parameters['timestep_parameter'] = np.asarray(0.0)
        self.default_dynamics_parameters['pid_iteration_time'] = np.asarray(0.)
        self.default_dynamics_parameters['mujoco_timestep'] = np.asarray(0.002)

        self.default_dynamics_parameters['action_additive_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['action_multiplicative_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['action_systematic_noise'] = np.asarray(0.0)

        self.default_dynamics_parameters['joint_obs_position_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['joint_obs_velocity_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['obs_position_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['obs_velocity_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['obs_angle_noise'] = np.asarray(0.0)

        self.default_dynamics_parameters['obj_density'] = np.asarray(400)
        self.default_dynamics_parameters['obj_size'] = np.array([0.0555 / 2, 0.0555 / 2, 0.03 / 2])
        self.default_dynamics_parameters['obj_sliding_friction'] = np.asarray(0.3)
        self.default_dynamics_parameters['obj_torsional_friction'] = np.asarray(0.01)

        link_masses = np.zeros((2,))
        for link_name, idx, body_node, mass_node, joint_node in self._robot_link_nodes_generator():
            if (mass_node is not None and idx in (5, 6)):
                dynamics_parameter_value = float(mass_node.get("mass"))
                link_masses[idx % 5] = dynamics_parameter_value

        self.default_dynamics_parameters['link_masses'] = link_masses
        self.default_dynamics_parameters['joint_dampings'] = self._default_damping_params()[-2:]
        self.default_dynamics_parameters['armatures'] = self._default_armature_params()[-2:]
        self.default_dynamics_parameters['joint_frictions'] = self._default_joint_friction_params()[-2:]

        if (use_pid):
            gains = self.mujoco_robot.velocity_pid_gains
            kps = np.array([gains['right_j{}'.format(actuator)]['p'] for actuator in (5, 6)])
            kis = np.array([gains['right_j{}'.format(actuator)]['i'] for actuator in (5, 6)])
            kds = np.array([gains['right_j{}'.format(actuator)]['d'] for actuator in (5, 6)])
            #
            self.default_dynamics_parameters['kps'] = kps
            self.default_dynamics_parameters['kis'] = kis
            self.default_dynamics_parameters['kds'] = kds
        else:
            kvs = np.zeros((2,))
            for target_joint, jnt_idx, node in self._velocity_actuator_nodes_generator():
                if jnt_idx in [5, 6]:
                    gains_value = float(node.get("kv"))
                    kvs[jnt_idx % 5] = gains_value

            self.default_dynamics_parameters['kvs'] = kvs

    def _set_default_parameter_sampling_ranges(self):
        """
        Returns the parameter ranges to draw samples from in the domain randomisation.
        """
        parameter_ranges = {
            'joint_forces': np.array([[0., 0.], [1.5, 1.5]]),  #
            'acceleration_forces': np.array([[0., 0., ], [0.05, 0.05, ]]),  #
            'obj_forces': np.array(
                [[0., 0., 0., 0., 0., 0., ], [0.00011, 0.00011, 0.00011, 0.00005, 0.00005, 0.00005, ]]),

            'joints_timedelay': np.array([0, 1]),
            'obj_timedelay': np.array([0, 2]),
            'timestep_parameter': np.array([0.0, 0.01]),
            'pid_iteration_time': np.array([0., 0.04]),
            'mujoco_timestep': np.array([0.001, 0.002]),

            'action_additive_noise': np.array([0.01, 0.1]),
            'action_multiplicative_noise': np.array([0.005, 0.02]),
            'action_systematic_noise': np.array([-0.05, 0.05]),

            'joint_obs_position_noise': np.array([0.0005, 0.005]),
            'joint_obs_velocity_noise': np.array([0.005, 0.005]),
            'obs_position_noise': np.array([0.0005, 0.001]),
            'obs_velocity_noise': np.array([0.0005, 0.0015]),
            'obs_angle_noise': np.array([0.005, 0.05]),

            'obj_density': np.array([100, 900]),
            'obj_size': np.array([0.995, 1.005]),
            'obj_sliding_friction': np.array([0.1, 0.85]),
            'obj_torsional_friction': np.array([0.001, 0.3]),

            'link_masses': np.array([0.98, 1.02]),
            'joint_dampings': np.array([0.5, 2.]),
            'armatures': np.array([0.66, 1.5]),
            'joint_frictions': np.array([0.66, 1.5]),
        }

        if (self.pid):
            parameter_ranges['kps'] = np.array([0.66, 1.5])
            parameter_ranges['kis'] = np.array([0.66, 1.5])
            parameter_ranges['kds'] = np.array([0.66, 1.5])
        else:
            parameter_ranges['kvs'] = [0.5, 2]

        self.parameter_sampling_ranges = parameter_ranges

    def _set_factors_for_param_randomisation(self, parameters):
        factors = copy.deepcopy(parameters)

        factors['joint_forces'] = np.ones((2,))
        factors['acceleration_forces'] = np.ones((2,))
        factors['obj_forces'] = np.ones((6,))

        factors['joints_timedelay'] = 1.0
        factors['timestep_parameter'] = 1.0
        factors['pid_iteration_time'] = 1.0
        factors['mujoco_timestep'] = 1.0
        factors['obj_timedelay'] = 1.0

        factors['action_additive_noise'] = 1.0
        factors['action_multiplicative_noise'] = 1.0
        factors['action_systematic_noise'] = 1.0

        factors['joint_obs_position_noise'] = 1.0
        factors['joint_obs_velocity_noise'] = 1.0
        factors['obs_position_noise'] = 1.0
        factors['obs_velocity_noise'] = 1.0
        factors['obs_angle_noise'] = 1.0

        factors['obj_density'] = 1.0
        factors['obj_sliding_friction'] = 1.0
        factors['obj_torsional_friction'] = 1.0

        self.factors_for_param_randomisation = factors

    def _velocity_actuator_nodes_generator(self):
        """
        Caching the xml nodes for the velocity actuators for use when setting the parameters
        """

        for node in self.model.root.findall(".//velocity[@kv]"):
            target_joint = node.get("joint")
            jnt_idx = int(target_joint[-1])
            yield target_joint, jnt_idx, node

    def _robot_link_nodes_generator(self):
        """
        Caching the xml nodes for the velocity actuators for use when setting the parameters
        """

        for link_idx, link_name in enumerate(self.mujoco_robot.links):
            body_node = self.mujoco_robot.root.find(".//body[@name='{}']".format(link_name))
            mass_node = body_node.find("./inertial[@mass]")
            joint_node = body_node.find("./joint")

            yield link_name, link_idx, body_node, mass_node, joint_node

    def _check_allowed_parameters(self, parameters):
        allowed_parameters = self.get_parameter_keys()

        for param in parameters:
            assert param in allowed_parameters, '{} not allowed. Only allowed parameters are {}'.format(param,
                                                                                                        allowed_parameters)

    def _select_appropriate_distribution(self, key):
        '''
        Which distribution to use to sample the different dynamics parameters.
        :param key: The parameter to consider.
        '''
        if (
                key == 'joint_forces'
                or key == 'acceleration_forces'
                or key == 'obj_forces'

                or key == 'timestep_parameter'
                or key == 'pid_iteration_time'
                or key == 'mujoco_timestep'

                or key == 'action_additive_noise'
                or key == 'action_multiplicative_noise'
                or key == 'action_systematic_noise'

                or key == 'joint_obs_position_noise'
                or key == 'joint_obs_velocity_noise'
                or key == 'obs_position_noise'
                or key == 'obs_velocity_noise'
                or key == 'obs_angle_noise'

                or key == 'link_masses'
                or key == 'obj_size'

                or key == 'obj_density'
                or key == 'obj_sliding_friction'
        ):
            return self.np_random.uniform
        elif (
                key == 'joints_timedelay'
                or key == 'obj_timedelay'
        ):
            return self._ranged_random_choice
        else:
            return self._loguniform

    def _loguniform(self, low=1e-10, high=1., size=None):
        return np.asarray(np.exp(self.np_random.uniform(np.log(low), np.log(high), size)))

    def _ranged_random_choice(self, low, high, size=1):
        vals = np.arange(low, high + 1)
        return self.np_random.choice(vals, size)

    def _parameter_for_randomisation_generator(self, parameters=None):
        '''
        Generates (key,value) pairs of sampled dynamics parameters.
         :param parameters: The parameters to be sampled for randomisation, if None, all the allowed parameters are sampled.
        '''
        parameter_ranges = self.parameter_sampling_ranges

        if (parameters is None):
            parameters = self.get_parameter_keys()

        for key in parameters:

            parameter_range = parameter_ranges[key]

            if (parameter_range.shape[0] == 1):
                yield key, np.asarray(parameter_range[0])
            elif (parameter_range.shape[0] == 2):
                distribution = self._select_appropriate_distribution(key)
                size = self.default_dynamics_parameters[key].shape
                yield key, np.asarray(
                    self.factors_for_param_randomisation[key] * distribution(*parameter_ranges[key], size=size))
            else:
                raise RuntimeError('Parameter radomisation range needs to be of shape {1,2}xN')

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model. This sets up the mujoco xml for the scene.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])
        obj_size = np.array([0.0555 / 2, 0.0555 / 2, 0.03 / 2])

        ### Domain Randomisation ###
        if (self.initialised):
            for key, val in self._parameter_for_randomisation_generator(parameters=self.parameters_to_randomise):
                self.dynamics_parameters[key] = val

            ## Queues for adding time delays
            self.goal_posemat_queue = deque(maxlen=int(self.dynamics_parameters['obj_timedelay'] + 1))
            self.obj_posemat_queue = deque(maxlen=int(self.dynamics_parameters['obj_timedelay'] + 1))
            self.obj_vel_queue = deque(maxlen=int(self.dynamics_parameters['obj_timedelay'] + 1))

            self.eef_qpos_queue = deque(maxlen=int(self.dynamics_parameters['joints_timedelay'] + 1))
            self.eef_qvel_queue = deque(maxlen=int(self.dynamics_parameters['joints_timedelay'] + 1))

            if (self.pid is not None):
                self.pid.sample_time = self.dynamics_parameters['pid_iteration_time']

            obj_size = self.dynamics_parameters['obj_size']

        ### Create the Task ###
        ## Load the Arena ##
        self.mujoco_arena = EmptyArena()

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        ## Create the objects that will go into the arena ##

        # Create object
        if (self.initialised):
            density = self.dynamics_parameters['obj_density']
            friction = np.array([self.dynamics_parameters['obj_sliding_friction'],
                                 self.dynamics_parameters['obj_torsional_friction'],
                                 1e-4])
        else:
            density = None
            friction = None

        rectangle = FullyFrictionalBoxObject(
            size_min=obj_size,
            size_max=obj_size,
            rgba=[1, 0, 0, 1],
            density=density,
            friction=friction
        )

        self.slide_object = rectangle

        ## Put everything together into the task ##
        self.model = SlideTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.slide_object,
        )

        # Add some small damping to the objects to prevent infinite acceleration
        self.model.xml_object.find('./joint').set('damping', '0.005')

        ### Set the xml parameters to the values given by the dynamics_parameters attribute ###
        if (self.initialised):
            self._apply_xml_dynamics_parameters()

    def _apply_xml_dynamics_parameters(self):
        """
        Applying the values contained in dynamics_parameters to the xml elements of the model. If a pid is used this
        also applied the pid gains contained in the dynamics parameters.
        """

        opt_node = self.model.root.find('option')
        opt_node.set("timestep", str(self.dynamics_parameters['mujoco_timestep']))

        joint_dampings = self._default_damping_params()
        joint_frictions = self._default_joint_friction_params()
        joint_armatures = self._default_armature_params()

        for link_name, idx, body_node, mass_node, joint_node in self._robot_link_nodes_generator():

            if (idx in (5, 6)):
                if (mass_node is not None):
                    mass_node.set("mass", str(self.dynamics_parameters['link_masses'][idx % 5]))

                if (joint_node is not None):
                    joint_node.set("damping", str(self.dynamics_parameters['joint_dampings'][idx % 5]))
                    joint_node.set("armature", str(self.dynamics_parameters['armatures'][idx % 5]))
                    joint_node.set("frictionloss", str(self.dynamics_parameters['joint_frictions'][idx % 5]))
            else:
                joint_node.set("damping", str(joint_dampings[idx]))
                joint_node.set("armature", str(joint_frictions[idx]))
                joint_node.set("frictionloss", str(joint_armatures[idx]))

        if (self.pid):
            kps = self.pid.Kp
            kis = self.pid.Ki
            kds = self.pid.Kd

            kps[-2:] = self.dynamics_parameters['kps']
            kis[-2:] = self.dynamics_parameters['kis']
            kds[-2:] = self.dynamics_parameters['kds']

            self.pid.tunings = (kps, kis, kds)
        else:
            for target_joint, jnt_idx, node in self._velocity_actuator_nodes_generator():
                if (jnt_idx in (5, 6)):
                    node.set("kv", str(self.dynamics_parameters['kvs'][jnt_idx % 5]))

    def set_parameter_sampling_ranges(self, sampling_ranges):
        '''
        Set a new sampling range for the dynamics parameters.
        :param sampling_ranges: (Dict) Dictionary of the sampling ranges for the different parameters of the form
        (param_name, range) where param_name is a valid param name string and range is a numpy array of dimensionality
        {1,2}xN where N is the dimension of the given parameter
        '''
        for candidate_name, candidate_value in sampling_ranges.items():
            assert candidate_name in self.parameter_sampling_ranges, 'Valid parameters are {}'.format(
                self.parameter_sampling_ranges.keys())
            assert candidate_value.shape[0] == 1 or candidate_value.shape[
                0] == 2, 'First dimension of the sampling parameter needs to have value 1 or 2'
            assert len(candidate_value.shape) == len(
                self.parameter_sampling_ranges[candidate_name].shape), '{} has the wrong number of dimensions'.format(
                candidate_name)
            if (len(self.parameter_sampling_ranges[candidate_name].shape) > 1):
                assert self.parameter_sampling_ranges[candidate_name].shape[1] == candidate_value.shape[
                    1], '{} has the wrong shape'.format(candidate_name)

            self.parameter_sampling_ranges[candidate_name] = candidate_value

    def get_parameter_sampling_ranges(self):
        return copy.deepcopy(self.parameter_sampling_ranges)

    def get_parameter_keys(self):
        return self.default_dynamics_parameters.keys()

    def get_total_parameter_dimension(self):
        total_dimension = 0
        for key, val in self.default_dynamics_parameters.items():
            param_shape = val.shape
            if (param_shape == ()):
                total_dimension += 1
            else:
                total_dimension += param_shape[0]
        return total_dimension

    def get_internal_state(self):
        return np.concatenate([self._joint_positions, self._joint_velocities]).tolist()

    def get_internal_state_dimension(self):
        internal_state = self.get_internal_state()
        return len(internal_state)

    def change_parameters_to_randomise(self, parameters):
        self._check_allowed_parameters(parameters)
        self._set_dynamics_parameters(self.default_dynamics_parameters)
        self.parameters_to_randomise = parameters

    def get_randomised_parameters(self):
        if (self.parameters_to_randomise is not None):
            return self.parameters_to_randomise
        else:
            return self.get_parameter_keys()

    def get_randomised_parameter_dimensions(self):
        """ Return the number of dimensions of the ranomised parameters"""
        randomised_parameter_names = self.get_randomised_parameters()

        total_dimension = 0
        for param in randomised_parameter_names:
            param_shape = self.default_dynamics_parameters[param].shape
            if (param_shape == ()):
                total_dimension += 1
            else:
                total_dimension += param_shape[0]
        return total_dimension

    def get_dynamics_parameters(self):
        """
        Returns the values of the current dynamics parameters.
        """
        return copy.deepcopy(self.dynamics_parameters)

    def get_default_dynamics_parameters(self):
        """
        Returns the default values of the  dynamics parameters.
        """
        return copy.deepcopy(self.default_dynamics_parameters)

    def get_factors_for_randomisation(self):
        """
        Returns the factor used for domain randomisation.
        """
        return copy.deepcopy(self.factors_for_param_randomisation)

    def set_dynamics_parameters(self, dynamics_parameter_dict):
        """
        Setting the dynamics parameters of the environment to specific values. These are going to be used the next
        time the environment is reset, and will be overriden if domain randomisation is on.
        :param dynamics_parameter_dict: Dictionary with the values of the parameters to set.
        """
        for key, value in dynamics_parameter_dict.items():
            assert key in self.dynamics_parameters, 'Setting a parameter that does not exist'
            self.dynamics_parameters[key] = value

    def randomisation_off(self, ):
        '''
        Disable the parameter randomisation temporarily and cache the current set of parameters and
        which parameters are being randomised.This can be useful for evaluation.
        '''
        current_params_to_randomise = self.get_randomised_parameters()
        current_params = self.get_dynamics_parameters()

        self.cached_parameters_to_randomise = current_params_to_randomise
        self.cached_dynamics_parameters = current_params

        self.parameters_to_randomise = []

        return current_params, current_params_to_randomise

    def randomisation_on(self):
        '''
        Restoring the randomisation as they were before the call to switch_params
        '''
        if (self.cached_dynamics_parameters is None):
            print("Randomisation was not switched off before switching it back on.")
            return

        self.parameters_to_randomise = self.cached_parameters_to_randomise
        self.set_dynamics_parameters(self.cached_dynamics_parameters)
        self.cached_parameters_to_randomise = None
        self.cached_dynamics_parameters = None

    def sample_parameter_randomisation(self, parameters=None):
        ''' Samples a dictionary of dynamics parameters values using the randomisation process currently set in the environment
            parameters ([string,]) : List of parameters to sample a randomisation from. If None, all the allowed parameters are sampled.
        '''
        if (not self.initialised):
            print('Function has undefined behaviour if environment fully initialised, returning with no effect')
            return

        parameters_sample = {}

        for key, val in self._parameter_for_randomisation_generator(parameters):
            assert key in self.get_parameter_keys(), '{} not allowed. Choose from {}'.format(key,
                                                                                             self.get_parameter_keys())
            parameters_sample[key] = val

        return parameters_sample

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Pushing object ids
        self.slide_obj_name = "slide_object"

        self.object_body_id = self.sim.model.body_name2id(self.slide_obj_name)
        self.object_geom_id = self.sim.model.geom_name2id(self.slide_obj_name)

        # Pushing object qpos indices for the object
        object_qpos = self.sim.model.get_joint_qpos_addr(self.slide_obj_name)
        self._ref_object_pos_low, self._ref_object_pos_high = object_qpos

        # Gripper ids
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        self.l_finger_body_id = self.sim.model.body_name2id('r_gripper_l_finger')
        self.r_finger_body_id = self.sim.model.body_name2id('r_gripper_r_finger')

        self.goal_site_id = self.sim.model.site_name2id('goal')

        self._ref_panel_body_indx = self.sim.model.body_name2id('slide_panel')

    def _reset_internal(self):
        """
        Resets simulation internal configurations. Is called upon environment reset.
        """
        super()._reset_internal()
        self.sim.forward()

        if (self.initialised):
            ### Set the arm position using IK ###

            ## Get the pose of the gripper in the initial position ##

            # Find the gripper length
            init_qpos = self.mujoco_robot.init_qpos + np.array([0., 0., 0., 0., 0., -np.pi / 2., 0.0])

            # Set the robot joint angles
            self.set_robot_joint_positions(init_qpos)

            # Set reference attributes
            self.init_qpos = init_qpos

            ### Set the object position  ###
            obj = self.model.slide_object
            obj_bottom_offset = obj.get_bottom_offset()
            start_site_pos = self.sim.data.get_site_xpos('start')

            if (self.randomise_initial_conditions):
                # bottom offset is negative and 0.0002 corrects for initial penetration
                obj_pos = np.array([start_site_pos[0], start_site_pos[1], start_site_pos[2] - obj_bottom_offset[2]- 0.0002])
                obj_pos += self.np_random.uniform(size=3) * np.array([0.002, 0.002, 0.0])

                # Get the object orientation
                obj_angle = self.np_random.uniform(-1, 1) * np.pi
                obj_quat = np.array([np.cos(obj_angle / 2), 0., 0., np.sin(obj_angle / 2)])
            else:
                # bottom offset is negative and 0.0002 corrects for initial penetration
                obj_pos = np.array([start_site_pos[0], start_site_pos[1], start_site_pos[2] - obj_bottom_offset[2]- 0.0002])
                obj_angle = -np.pi / 2.
                obj_quat = np.array([np.cos(obj_angle / 2), 0., 0., np.sin(obj_angle / 2)])

            # Concatenate to get the object qpos
            obj_qpos = np.concatenate([obj_pos, obj_quat])
            #
            self.sim.data.qpos[self._ref_object_pos_low:self._ref_object_pos_high] = obj_qpos
            self.sim.forward()

            self.correct_for_external_forces()

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has three components.

            Reaching: in [-inf, 0], to encourage the arm to reach the object
            Goal Distance: in [-inf, 0] the distance between the pushed object and the goal
            Safety reward in [-inf, 0], -1 for every joint that is at its limit.

        The sparse reward only receives a {0,1} upon reaching the goal

        Args:
            action (np array): The action taken in that timestep

        Returns:
            reward (float or dict): the reward if sparce rewards are used otherwise a dictionary
            with the total reward, and the subcoponents of the dense reward.
        """
        reward = 0.

        # sparse completion reward
        if not self.reward_shaping and self._check_success():
            reward = 1.0

        # use a dense reward
        if self.reward_shaping:
            object_pos = self.sim.data.body_xpos[self.object_body_id]

            ## Hitting limits reward
            joint_limits = self._joint_ranges
            current_joint_pos = self._joint_positions

            hitting_limits_reward = - int(
                any([(x < joint_limits[i, 0] + 0.03 or x > joint_limits[i, 1] - 0.03) for i, x in
                     enumerate(current_joint_pos)]))

            reward += hitting_limits_reward

            # # Success Reward
            success = self._check_success()
            success_reward = 0.0
            if (success):
                success_reward += 0.1

            reward += success_reward

            # # goal distance reward
            goal_pos = self.sim.data.site_xpos[self.goal_site_id]

            obj_goal_in_world = goal_pos - object_pos

            dist = np.linalg.norm(obj_goal_in_world)
            goal_distance_reward = - dist
            reward += goal_distance_reward

            fallen_object_reward = 0.0
            fallen_object = np.abs(obj_goal_in_world[2]) > 0.5
            if (fallen_object):
                fallen_object_reward = -0.5

            reward += fallen_object_reward

            # Return all three types of rewards
            reward = {"reward": reward, "reaching_distance": 0.0,
                      "goal_distance": goal_distance_reward,
                      "success_reward": success_reward,
                      "hitting_limits_reward": hitting_limits_reward,
                      "unstable": False,
                      "fallen_object_reward": fallen_object_reward,
                      "fallen_object": fallen_object}

        return reward

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        object_pos = self.sim.data.body_xpos[self.object_body_id]
        goal_pos = self.sim.data.site_xpos[self.goal_site_id]

        dist = np.linalg.norm(goal_pos - object_pos)

        if (dist <= 0.023):
            return True

        return False

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        return 2

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
            # maximum velocity command to 0.3 rads/sec
            ctrl_range_low, ctrl_range_high = self._joint_velocities_range()
            ctrl_range_low = ctrl_range_low[-2:]
            ctrl_range_high = ctrl_range_high[-2:]
            return ctrl_range_low, ctrl_range_high

    def _pre_action(self, action):
        """ Takes the action, randomised the control timestep, and adds some additional random noise to the action."""
        assert len(action) == self.dof, "environment got invalid action dimension"
        low, high = self.action_spec

        # Change control timestep to simulate various random time delays
        timestep_parameter = self.dynamics_parameters['timestep_parameter']
        self.control_timestep = self.init_control_timestep + self.np_random.exponential(scale=timestep_parameter)

        # Add action noise to simulate unmodelled effects
        action_range = high - low
        additive_noise = action_range * self.dynamics_parameters[
            'action_additive_noise'] * self.np_random.uniform(-1, 1, action.shape)
        additive_systematic_noise = action_range * self.dynamics_parameters['action_systematic_noise']
        multiplicative_noise = 1.0 + (
                self.dynamics_parameters['action_multiplicative_noise'] * self.np_random.uniform(-1, 1,
                                                                                                 action.shape))

        action = action * multiplicative_noise + additive_noise + additive_systematic_noise

        action = np.clip(action, low, high)

        # Deal with the arm action
        if (self.normalised_actions):
            # Set the maximum velocity command to 0.3rads/sec
            ctrl_range_low, ctrl_range_high = self._joint_velocities_range()
            ctrl_range_low = ctrl_range_low[-2:]
            ctrl_range_high = ctrl_range_high[-2:]
            bias = 0.5 * (ctrl_range_high + ctrl_range_low)
            weight = 0.5 * (ctrl_range_high - ctrl_range_low)
            action = bias + weight * action

        if (self.pid is None):
            self.sim.data.ctrl[self.mujoco_robot.dof - 2:self.mujoco_robot.dof] = action
        else:
            self.pid.reset()
            self.pid.setpoint = np.concatenate([np.zeros((5,)), action])

        self.joint_random_forces = self.dynamics_parameters['joint_forces'] * self.np_random.uniform(-1, 1, 2)
        self.joint_acceleration_forces = self.dynamics_parameters['acceleration_forces'] * \
                                         self.sim.data.qacc[self._ref_joint_vel_indexes][-2:]
        self.object_random_forces = self.dynamics_parameters['obj_forces'] * self.np_random.uniform(-1, 1, 6)

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        self._pre_action(action)
        end_time = self.cur_time + self.control_timestep
        while self.cur_time < end_time:
            self.correct_for_external_forces()
            if (self.pid is not None):
                self._set_pid_control()
            self.sim.step()
            self.cur_time += self.model_timestep
        reward, done, info = self._post_action(action)
        return self._get_observation(), reward, done, info

    def _set_pid_control(self):
        dt = self.model_timestep if self.pid._last_output is not None else 1e-16
        current_qvel = copy.deepcopy(self.sim.data.actuator_velocity)
        velocity_feedback_control = self.pid(current_qvel, dt)
        self.sim.data.ctrl[:] = velocity_feedback_control

    def _post_action(self, action):
        """
        Add dense reward subcomponents to info, and checks for success of the task.
        """
        reward, done, info = super()._post_action(action)

        if self.reward_shaping:
            info = reward
            reward = reward["reward"]

        if (info["fallen_object"]):
            done = True

        if (info["unstable"]):
            done = True

        info["success"] = self._check_success()

        return reward, done, info

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            gripper_to_object : The x-y component of the gripper to object distance
            object_to_goal : The x-y component of the object-to-goal distance
            object_z_rot : the roation of the object around an axis sticking out the table

            object_xvelp: x-y linear velocity of the object
            gripper_xvelp: x-y linear velocity of the gripper


            task-state : a concatenation of all the above.
        """
        di = OrderedDict()

        slide_object_name = "slide_object"
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # Extract goal position and rotation
            goal_pos_in_world = self.sim.data.get_site_xpos('goal')
            goal_rot_in_world = self.sim.data.get_site_xmat('goal')

            goal_posemat_in_world = T.make_pose(goal_pos_in_world, goal_rot_in_world)
            # Correct for the fact that the goal site is in the middle of the slide plank
            goal_posemat_in_world = goal_posemat_in_world.dot(np.array([[1., 0., 0., 0.],
                                                                        [0., 1., 0., 0.],
                                                                        [0., 0., 1., 0.002],
                                                                        [0., 0., 0., 1.]]))

            # Apply time delays
            goal_posemat_in_world = self._apply_time_delay(goal_posemat_in_world, self.goal_posemat_queue)

            # Add random noise
            goal_pos_noise = self.dynamics_parameters['obs_position_noise']
            goal_posemat_in_world[:3, 3] = goal_posemat_in_world[:3, 3] + self.np_random.normal(loc=0.,
                                                                                                scale=goal_pos_noise)

            goal_angle_noise = self.dynamics_parameters['obs_angle_noise']
            goal_angle_x_noise = self.np_random.normal(loc=0., scale=goal_angle_noise)
            goal_angle_y_noise = self.np_random.normal(loc=0., scale=goal_angle_noise)
            goal_angle_z_noise = self.np_random.normal(loc=0., scale=goal_angle_noise)

            goal_rot_noise = T.euler2mat(goal_angle_x_noise, goal_angle_y_noise, goal_angle_z_noise)

            goal_posemat_in_world[:3, :3] = goal_rot_noise.dot(goal_posemat_in_world[:3, :3])

            # Extract object position , velocity and roation
            object_pos_in_world = self.sim.data.get_body_xpos(slide_object_name)
            object_rot_in_world = self.sim.data.get_body_xmat(slide_object_name)
            object_vel_in_world = self.sim.data.get_body_xvelp(slide_object_name)

            object_posemat_in_world = T.make_pose(object_pos_in_world, object_rot_in_world)

            # Apply time delays
            object_posemat_in_world = self._apply_time_delay(object_posemat_in_world, self.obj_posemat_queue)
            object_vel_in_world = self._apply_time_delay(object_vel_in_world, self.obj_vel_queue)

            # Add random noise
            object_pos_noise = self.dynamics_parameters['obs_position_noise']
            object_posemat_in_world[:3, 3] = object_posemat_in_world[:3, 3] + self.np_random.normal(loc=0.,
                                                                                                    scale=object_pos_noise)

            object_angle_noise = self.dynamics_parameters['obs_angle_noise']
            object_angle_x_noise = self.np_random.normal(loc=0., scale=object_angle_noise)
            object_angle_y_noise = self.np_random.normal(loc=0., scale=object_angle_noise)
            object_angle_z_noise = self.np_random.normal(loc=0., scale=object_angle_noise)

            object_rot_noise = T.euler2mat(object_angle_x_noise, object_angle_y_noise, object_angle_z_noise)

            object_posemat_in_world[:3, :3] = object_rot_noise.dot(object_posemat_in_world[:3, :3])

            object_vel_noise = self.dynamics_parameters['obs_velocity_noise']
            object_vel_in_world = object_vel_in_world + self.np_random.normal(loc=0., scale=object_vel_noise)

            # Extract the joint poition and velocity
            qpos = self.sim.data.qpos[self._ref_joint_pos_indexes][-2:]
            qvel = self.sim.data.qvel[self._ref_joint_vel_indexes][-2:]

            # Apply time delays
            qpos = self._apply_time_delay(qpos, self.eef_qpos_queue)
            qvel = self._apply_time_delay(qvel, self.eef_qvel_queue)

            # Add random noise
            qpos_noise = self.dynamics_parameters['joint_obs_position_noise']
            qvel_noise = self.dynamics_parameters['joint_obs_velocity_noise']

            qpos = qpos + self.np_random.normal(loc=0., scale=qpos_noise)
            qvel = qvel + self.np_random.normal(loc=0., scale=qvel_noise)

            # Convert reference frames and construct observation
            world_posemat_in_goal = T.pose_inv(goal_posemat_in_world)
            world_rot_in_goal = world_posemat_in_goal[:3, :3]
            object_posemat_in_goal = T.pose_in_A_to_pose_in_B(object_posemat_in_world, world_posemat_in_goal)

            object_pos_in_goal = object_posemat_in_goal[:3, 3]
            object_orn_in_goal = T.mat2quat(object_posemat_in_goal[:3, :3])

            z_angle = T.mat2euler(object_posemat_in_goal[:3, :3])[2]
            sin_cos = np.array([np.sin(8 * z_angle), np.cos(8 * z_angle)])

            object_vel_in_goal = world_rot_in_goal.dot(object_vel_in_world)

            di['object_pos_in_goal'] = object_pos_in_goal
            di['object_orn_in_goal'] = object_orn_in_goal
            di['object_vel_in_goal'] = object_vel_in_goal
            di['qpos'] = qpos
            di['qvel'] = qvel
            di['z_angle'] = z_angle
            di['sin_cos'] = sin_cos

            di["task-state"] = np.concatenate(
                [object_pos_in_goal, sin_cos,
                 object_vel_in_goal, qpos,
                 qvel]
            )

        return di

    def _apply_time_delay(self, object, queue):
        queue.appendleft(copy.deepcopy(object))

        if (len(queue) == queue.maxlen):
            return queue.pop()
        else:
            return queue[-1]

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                    self.sim.model.geom_id2name(contact.geom1)
                    in self.gripper.contact_geoms()
                    or self.sim.model.geom_id2name(contact.geom2)
                    in self.gripper.contact_geoms()
            ):
                collision = True
                break
        return collision

    def _check_contact_with(self, object):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if (
                    (self.sim.model.geom_id2name(contact.geom1) in self.gripper.contact_geoms()
                     and contact.geom2 == self.sim.model.geom_name2id(object))

                    or (self.sim.model.geom_id2name(contact.geom2) in self.gripper.contact_geoms()
                        and contact.geom1 == self.sim.model.geom_name2id(object))
            ):
                print(i, contact.frame)
                print(contact.frame.reshape(3, 3))
                print('dist', contact.dist)
                print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
                print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))

                c_array = np.zeros(6, dtype=np.float64)
                print('c_array', c_array)
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
                print('c_array', c_array)

                collision = True
                break
        return collision

    def correct_for_external_forces(self):
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

        # Correct for contact forces
        forces, points, bodies = self.get_contact_forces_with_eef()

        for i, point in enumerate(points):
            generalised_force = np.zeros(14, dtype=np.float64)

            mujoco_py.functions.mj_applyFT(self.sim.model, self.sim.data,
                                           forces[i][:3], forces[i][3:], point, self.sim.model.body_name2id(bodies[i]),
                                           generalised_force)

            self.sim.data.qfrc_applied[:] -= copy.deepcopy(generalised_force)

        # Domain randomisation forces
        #  Adding  forces to the joint
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes[-2:]
        ] += self.joint_random_forces

        # Adding force proportional to acceleration
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes[-2:]
        ] += self.joint_acceleration_forces

        # Adding forces to the object
        obj_qvel_low_idx, obj_qvel_high_idx = self.sim.model.get_joint_qvel_addr('slide_object')
        self.sim.data.qfrc_applied[
        obj_qvel_low_idx: obj_qvel_high_idx
        ] += self.object_random_forces

    def get_contact_forces_with_eef(self):
        total_generalised_forces = []
        application_points = []
        bodies = []

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            contact_geom = None
            if (self.sim.model.geom_id2name(contact.geom1) in self.gripper.contact_geoms()):
                contact_geom = contact.geom1

            if (self.sim.model.geom_id2name(contact.geom2) in self.gripper.contact_geoms()):
                contact_geom = contact.geom2

            if (contact_geom is not None):
                body_name = self.gripper.get_contact_body_from_geom(self.sim.model.geom_id2name(contact_geom))
                bodies.append(body_name)

                total_generalised_force = np.zeros(6, dtype=np.float64)
                c_array = np.zeros(6, dtype=np.float64)
                frame = contact.frame.reshape((3, 3))
                application_points.append(contact.pos)

                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)

                total_generalised_force[:3] += frame.dot(c_array[:3])
                total_generalised_force[3:] += frame.dot(c_array[3:])
                total_generalised_forces.append(total_generalised_force)

        return total_generalised_forces, application_points, bodies

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to object
        if self.gripper_visualization:
            # get distance to object
            object_site_id = self.sim.model.site_name2id("slide_object")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[object_site_id]
                    - self.sim.data.get_site_xpos("grip_site")
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
