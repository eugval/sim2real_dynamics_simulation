from collections import OrderedDict
import numpy as np
from robosuite_extra.env_base.sawyer import SawyerEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite_extra.models.generated_objects import FullyFrictionalBoxObject
from robosuite_extra.models.tasks import UniformSelectiveSampler
from robosuite.utils.mjcf_utils import array_to_string
from robosuite_extra.push_env.push_task import PushTask
from robosuite_extra.utils import transform_utils as T
from robosuite_extra.controllers import SawyerEEFVelocityController
import copy
from collections import deque

class SawyerPush(SawyerEnv):
    """
    This class corresponds to a Pushing task for the sawyer robot arm.

    This task consists of pushing a rectangular puck from some initial position to a final goal.
    The goal and initial positions are chosen randomly within some starting bounds
    """

    def __init__(
            self,
            gripper_type="PushingGripper",
            parameters_to_randomise=None,
            randomise_initial_conditions=True,
            table_full_size=(0.8, 1.6, 0.719),
            table_friction=(1e-4, 5e-3, 1e-4),
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            placement_initializer=None,
            gripper_visualization=True,
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

            parameters_to_randomise [string,] : List of keys for parameters to randomise, None means all the available parameters are randomised


            randomise_initial_conditions [bool,]: Whether or not to randomise the starting configuration of the task.


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

        # settings for table
        self.table_full_size = table_full_size
        self.table_friction = table_friction

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


        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformSelectiveSampler(
                x_range=None,
                y_range=None,
                ensure_object_boundary_in_range=True,
                z_rotation=None,
                np_random=None
            )

            # Param for storing a specific goal  and object starting positions
            self.specific_goal_position = None
            self.specific_gripper_position = None
            self.gripper_pos_neutral = [0.44969246, 0.16029991, 1.00875409]

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
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

        self.placement_initializer.set_random_number_generator(self.np_random)

        self.init_control_timestep = self.control_timestep
        self.init_qpos = self.mujoco_robot.init_qpos

        # Storing  parameters for temporary switching
        self.cached_parameters_to_randomise = None
        self.cached_dynamics_parameters = None


        self.initialised = True
        self.reset()

    def _set_dynamics_parameters(self, parameters):
        self.dynamics_parameters = copy.deepcopy(parameters)

    def _default_damping_params(self):
        # return np.array([0.01566, 1.171, 0.4906, 0.1573, 1.293, 0.08688, 0.1942]) # -real world calibration
        # return np.array([0.8824,2.3357,1.1729, 0.0 , 0.5894, 0.0  ,0.0082]) #- command calibration
        return np.array([8.19520686e-01, 1.25425414e+00, 1.04222253e+00,
                         0.00000000e+00, 1.43146116e+00, 1.26807887e-01, 1.53680244e-01, ])  # - command calibration 2

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
        self.default_dynamics_parameters['joint_forces'] = np.zeros((7,))
        self.default_dynamics_parameters['acceleration_forces'] = np.zeros((7,))
        self.default_dynamics_parameters['eef_forces'] = np.zeros((6,))
        self.default_dynamics_parameters['obj_forces'] = np.zeros((6,))


        self.default_dynamics_parameters['eef_timedelay'] = np.asarray(0)
        self.default_dynamics_parameters['obj_timedelay'] = np.asarray(0)
        self.default_dynamics_parameters['timestep_parameter'] = np.asarray(0.0)
        self.default_dynamics_parameters['pid_iteration_time'] = np.asarray(0.)
        self.default_dynamics_parameters['mujoco_timestep'] = np.asarray(0.002)

        self.default_dynamics_parameters['action_additive_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['action_multiplicative_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['action_systematic_noise'] = np.asarray(0.0)

        self.default_dynamics_parameters['eef_obs_position_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['eef_obs_velocity_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['obj_obs_position_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['obj_obs_velocity_noise'] = np.asarray(0.0)
        self.default_dynamics_parameters['obj_angle_noise'] = np.asarray(0.0)

        self.default_dynamics_parameters['obj_density'] = np.asarray(400)
        self.default_dynamics_parameters['obj_size'] =  np.array([0.0555 / 2, 0.0555 / 2, 0.03 / 2])
        self.default_dynamics_parameters['obj_sliding_friction'] = np.asarray(0.4)
        self.default_dynamics_parameters['obj_torsional_friction'] = np.asarray(0.01)

        link_masses = np.zeros((7,))
        for link_name, idx, body_node, mass_node, joint_node in self._robot_link_nodes_generator():
            if (mass_node is not None):
                dynamics_parameter_value = float(mass_node.get("mass"))
                link_masses[idx] = dynamics_parameter_value

        self.default_dynamics_parameters['link_masses'] = link_masses
        self.default_dynamics_parameters['joint_dampings'] = self._default_damping_params()
        self.default_dynamics_parameters['armatures'] = self._default_armature_params()
        self.default_dynamics_parameters['joint_frictions'] = self._default_joint_friction_params()

        if (use_pid):
            gains = self.mujoco_robot.velocity_pid_gains
            kps = np.array([gains['right_j{}'.format(actuator)]['p'] for actuator in range(7)])
            kis = np.array([gains['right_j{}'.format(actuator)]['i'] for actuator in range(7)])
            kds = np.array([gains['right_j{}'.format(actuator)]['d'] for actuator in range(7)])
            #
            self.default_dynamics_parameters['kps'] = kps
            self.default_dynamics_parameters['kis'] = kis
            self.default_dynamics_parameters['kds'] = kds
        else:
            kvs = np.zeros((7,))
            for target_joint, jnt_idx, node in self._velocity_actuator_nodes_generator():
                gains_value = float(node.get("kv"))
                kvs[jnt_idx] = gains_value

            self.default_dynamics_parameters['kvs'] = kvs

    def _set_default_parameter_sampling_ranges(self):
        """
        Returns the parameter ranges to draw samples from in the domain randomisation.
        """
        parameter_ranges = {
            'joint_forces': np.array([[0.,0.,0.,0.,0.,0.,0.], [1.5,1.5,1.5,1.5,1.5,1.5,1.5]]),#
            'acceleration_forces': np.array([[0.,0.,0.,0.,0.,0.,0.], [0.05,0.05,0.05,0.05,0.05,0.05,0.05]]),#
            'eef_forces': np.array([[0.,0.,0.,0.,0.,0.], [0.06 ,0.06,0.06,0.01,0.01,0.01,]]), #
            'obj_forces': np.array([[0., 0., 0., 0., 0., 0., ], [0.0011, 0.0011, 0.0011, 0.0005, 0.0005, 0.0005, ]]),

            'eef_timedelay': np.array([0, 1]),
            'obj_timedelay': np.array([0,2]),
            'timestep_parameter': np.array([0.0, 0.01]),
            'pid_iteration_time':  np.array([0., 0.04]),
            'mujoco_timestep': np.array([0.001,0.002]),

            'action_additive_noise': np.array([0.01, 0.1]),
            'action_multiplicative_noise': np.array([0.005,0.02]),
            'action_systematic_noise': np.array([-0.05, 0.05]),

            'eef_obs_position_noise': np.array([0.0005, 0.001]),
            'eef_obs_velocity_noise': np.array([0.0005, 0.001]),
            'obj_obs_position_noise': np.array([0.0005, 0.001]),
            'obj_obs_velocity_noise': np.array([0.0005, 0.0015]),
            'obj_angle_noise': np.array([0.005, 0.05]),

            'obj_density': np.array([100, 800]),
            'obj_size': np.array([0.995, 1.005]),
            'obj_sliding_friction': np.array([0.01, 0.8]),
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

        factors['joint_forces'] = np.ones((7,))
        factors['acceleration_forces'] = np.ones((7,))
        factors['eef_forces'] = np.ones((1,))
        factors['obj_forces'] = np.ones((6,))

        factors['eef_timedelay'] = 1.0
        factors['timestep_parameter'] = 1.0
        factors['pid_iteration_time'] = 1.0
        factors['mujoco_timestep'] = 1.0
        factors['obj_timedelay'] = 1.0

        factors['action_additive_noise'] = 1.0
        factors['action_multiplicative_noise'] = 1.0
        factors['action_systematic_noise'] = 1.0

        factors['eef_obs_position_noise'] = 1.0
        factors['eef_obs_velocity_noise'] = 1.0
        factors['obj_obs_position_noise'] = 1.0
        factors['obj_obs_velocity_noise'] = 1.0
        factors['obj_angle_noise'] = 1.0

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
                or key == 'eef_forces'
                or key == 'obj_forces'

                or key == 'timestep_parameter'
                or key == 'pid_iteration_time'
                or key == 'mujoco_timestep'

                or key == 'action_additive_noise'
                or key == 'action_multiplicative_noise'
                or key == 'action_systematic_noise'

                or key == 'eef_obs_position_noise'
                or key == 'eef_obs_velocity_noise'
                or key == 'obj_obs_position_noise'
                or key == 'obj_obs_velocity_noise'
                or key == 'obj_angle_noise'

                or key == 'link_masses'
                or key == 'obj_size'

                or key == 'obj_density'
                or key == 'obj_sliding_friction'
            ):
            return self.np_random.uniform
        elif (
                key == 'eef_timedelay'
                or key == 'obj_timedelay'
        ):
            return self._ranged_random_choice
        else:
            return self._loguniform

    def _loguniform(self, low=1e-10, high=1., size=None):
        return np.asarray(np.exp(self.np_random.uniform(np.log(low), np.log(high), size)))

    def _ranged_random_choice(self,low, high, size=1):
        vals = np.arange(low,high+1)
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
            self.eef_pos_queue = deque(maxlen=int(self.dynamics_parameters['eef_timedelay'] + 1))
            self.eef_vel_queue = deque(maxlen=int(self.dynamics_parameters['eef_timedelay'] + 1))

            self.obj_pos_queue = deque(maxlen=int(self.dynamics_parameters['obj_timedelay'] + 1))
            self.obj_vel_queue = deque(maxlen=int(self.dynamics_parameters['obj_timedelay'] + 1))
            self.obj_angle_queue = deque(maxlen=int(self.dynamics_parameters['obj_timedelay'] + 1))

            if (self.pid is not None):
                self.pid.sample_time = self.dynamics_parameters['pid_iteration_time']

            obj_size = self.dynamics_parameters['obj_size']

        ### Create the Task ###
        ## Load the Arena ##
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        ## Create the objects that will go into the arena ##
        # Create object and goal

        if(self.initialised):
            density = self.dynamics_parameters['obj_density']
            friction = np.array([self.dynamics_parameters['obj_sliding_friction'],
                               self.dynamics_parameters['obj_torsional_friction'],
                               self.table_friction[2]])
        else:
            density = None
            friction = None

        rectangle = FullyFrictionalBoxObject(
            size_min= obj_size,  #
            size_max=  obj_size,  #
            rgba=[1, 0, 0, 1],
            density=density,
            friction=friction
        )


        self.mujoco_objects = OrderedDict([("rectangle", rectangle)])


        goal = CylinderObject(
            size=[0.03, 0.001],
            rgba=[0, 1, 0, 1],
        )
        self.mujoco_goal = goal

        ## Put everything together into the task ##
        self.model = PushTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_goal,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )

        # Add some small damping to the objects to prevent infinite acceleration
        for obj in self.model.xml_objects:
            obj.find('./joint').set('damping', '0.005')

        ## Manipulate objects in task ##
        # Reduce penetration of objects
        for obj in self.model.xml_objects:
            obj.find('geom').set('solimp', "0.99 0.99 0.01")
            obj.find('geom').set('solref', "0.01 1")

        self.model.arena.table_collision.set('solimp', "0.99 0.99 0.01")
        self.model.arena.table_collision.set('solref', "0.01 1")

        # Place goal: it can be placed anywhere in a 16x30 cm box centered 15 cm away
        # from the center of the table along its length
        if (self.specific_goal_position is not None):
            g_pos = np.array([self.gripper_pos_neutral[0] + self.specific_goal_position[0],
                                 self.gripper_pos_neutral[1] + self.specific_goal_position[1],
                                 self.model.table_top_offset[2]])

        elif (self.randomise_initial_conditions):
            noise = self.np_random.uniform(-1, 1, 3) * np.array([0.15, 0.08, 0.0])
            offset = np.array([0.0, 0.15, 0.0])
            g_pos = noise + offset + self.model.table_top_offset
        else:
            g_pos = [0.44969246, 0.16029991 + 0.335, self.model.table_top_offset[2]] #Placing the object at 30 cm , the eef needs to be at 33.5 cm

        self.model.xml_goal.set("pos", array_to_string(g_pos))

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

        for link_name, idx, body_node, mass_node, joint_node in self._robot_link_nodes_generator():
            if (mass_node is not None):
                mass_node.set("mass", str(self.dynamics_parameters['link_masses'][idx]))

            if (joint_node is not None):
                joint_node.set("damping", str(self.dynamics_parameters['joint_dampings'][idx]))
                joint_node.set("armature", str(self.dynamics_parameters['armatures'][idx]))
                joint_node.set("frictionloss", str(self.dynamics_parameters['joint_frictions'][idx]))

        if (self.pid):
            self.pid.tunings = (self.dynamics_parameters['kps'],
                                self.dynamics_parameters['kis'],
                                self.dynamics_parameters['kds'],
                                )
        else:
            for target_joint, jnt_idx, node in self._velocity_actuator_nodes_generator():
                node.set("kv", str(self.dynamics_parameters['kvs'][jnt_idx]))




    def set_parameter_sampling_ranges(self, sampling_ranges):
        '''
        Set a new sampling range for the dynamics parameters.
        :param sampling_ranges: (Dict) Dictionary of the sampling ranges for the different parameters of the form
        (param_name, range) where param_name is a valid param name string and range is a numpy array of dimensionality
        {1,2}xN where N is the dimension of the given parameter
        '''
        for candidate_name, candidate_value in sampling_ranges.items():
            assert candidate_name in  self.parameter_sampling_ranges, 'Valid parameters are {}'.format(self.parameter_sampling_ranges.keys())
            assert candidate_value.shape[0] == 1 or candidate_value.shape[0]==2, 'First dimension of the sampling parameter needs to have value 1 or 2'
            assert len(candidate_value.shape) == len(self.parameter_sampling_ranges[candidate_name].shape), '{} has the wrong number of dimensions'.format(candidate_name)
            if(len(self.parameter_sampling_ranges[candidate_name].shape) >1):
                assert self.parameter_sampling_ranges[candidate_name].shape[1] == candidate_value.shape[1], '{} has the wrong shape'.format(candidate_name)

            self.parameter_sampling_ranges[candidate_name] = candidate_value

    def get_parameter_sampling_ranges(self):
        return copy.deepcopy(self.parameter_sampling_ranges)

    def get_parameter_keys(self):
        return self.default_dynamics_parameters.keys()

    def get_total_parameter_dimension(self):
        total_dimension = 0
        for key, val in self.default_dynamics_parameters.items():
            param_shape = val.shape
            if(param_shape ==()):
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
            if(param_shape ==()):
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

    def randomisation_off(self,):
        '''
        Disable the parameter randomisation temporarily and cache the current set of parameters and
        which parameters are being randomised.This can be useful for evaluation.
        '''
        current_params_to_randomise = self.get_randomised_parameters()
        current_params = self.get_dynamics_parameters()

        self.cached_parameters_to_randomise = current_params_to_randomise
        self.cached_dynamics_parameters = current_params

        self.parameters_to_randomise = []

        return current_params,  current_params_to_randomise

    def randomisation_on(self):
        '''
        Restoring the randomisation as they were before the call to switch_params
        '''
        if(self.cached_dynamics_parameters is None):
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

    def _set_goal_neutral_offset(self, goal_x, goal_y):
        self.specific_goal_position =  np.array([goal_x, goal_y])

    def _set_gripper_neutral_offset(self, gripper_x, gripper_y):
        self.specific_gripper_position =  np.array([gripper_x, gripper_y])

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Pushing object ids
        self.push_obj_name = self.model.object_names[self.model.push_object_idx]

        self.object_body_id = self.sim.model.body_name2id(self.push_obj_name)
        self.object_geom_id = self.sim.model.geom_name2id(self.push_obj_name)

        # Pushing object qpos indices for the object
        object_qpos = self.sim.model.get_joint_qpos_addr(self.push_obj_name)
        self._ref_object_pos_low, self._ref_object_pos_high = object_qpos

        # goal ids
        self.goal_body_id = self.sim.model.body_name2id("goal")
        self.goal_site_id = self.sim.model.site_name2id("goal")

        # Gripper ids
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

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
            gripper_site = self.sim.data.site_xpos[self.eef_site_id]
            right_hand_pos = self.sim.data.get_body_xpos('right_hand')
            gripper_length = (right_hand_pos - gripper_site)[2]

            if(self.specific_gripper_position is not None):
                init_pos = np.array([self.gripper_pos_neutral[0] + self.specific_gripper_position[0],
                                     self.gripper_pos_neutral[1] + self.specific_gripper_position[1],
                                     self.model.table_top_offset.copy()[2] + 0.007+ gripper_length])


                init_pose = T.make_pose(init_pos, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))

            elif (self.randomise_initial_conditions):
                # Get the initial position of interest :
                # A box of size 12x12cm, 15 cm away from the center of the table in the y axis
                noise = self.np_random.uniform(-1, 1, 3) * np.array([0.12, 0.12, 0.0])
                offset = np.array([0.0, -0.15, 0.007])
                init_pos = self.model.table_top_offset.copy() + noise + offset
                init_pos[2] = init_pos[2] + gripper_length
                init_pose = T.make_pose(init_pos, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))  #
            else:
                gripper_pos = self.sim.data.get_site_xpos('grip_site')
                init_pos = np.concatenate([gripper_pos[:2], [self.model.table_top_offset.copy()[2] + 0.007]])
                init_pos[2] = init_pos[2] + gripper_length
                init_pose = T.make_pose(init_pos, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))

            ## Do the IK to find the joint angles for this initial pose ##
            # Start the IK search from the rest qpos
            ref_q = self.mujoco_robot.init_qpos

            # Express init_pose in the base frame of the robot
            init_pose_in_base = self.pose_in_base(init_pose)

            # Do the IK
            joint_angles = self.IK_solver.compute_joint_angles_for_endpoint_pose(init_pose_in_base, ref_q)

            # Set the robot joint angles
            self.set_robot_joint_positions(joint_angles)

            # Set reference attributes
            self.init_qpos = joint_angles
            self.init_right_hand_quat = self._right_hand_quat
            self.init_right_hand_orn = self._right_hand_orn
            self.init_right_hand_pos = self._right_hand_pos

            eef_rot_in_world = self.sim.data.get_body_xmat("right_hand").reshape((3, 3))
            self.world_rot_in_eef = copy.deepcopy(eef_rot_in_world.T)

            ### Set the object position next to the arm ###

            # Find End effector position
            eef_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])

            # Get the mujoco pusing object
            obj = self.model.mujoco_objects[self.model.push_object_idx]

            # Find the position just next to the eef
            obj_radius = obj.get_horizontal_radius()
            obj_bottom_offset = obj.get_bottom_offset()
            if (self.randomise_initial_conditions):
                obj_pos = np.array([eef_pos[0], eef_pos[1] + obj_radius + 0.00701,
                                    self.model.table_top_offset[2] - obj_bottom_offset[2]])
                obj_pos += self.np_random.uniform(size=3) * np.array([0.0012, 0.001, 0.0])

                # Get the object orientation
                obj_angle = np.pi / 2. + self.np_random.uniform(-1, 1) * np.pi / 6.
                obj_quat = np.array([np.cos(obj_angle / 2.), 0., 0., np.sin(obj_angle / 2.)])
            else:
                obj_pos = np.array([eef_pos[0], eef_pos[1] + obj.size[0] +0.0071+ 0.0002 ,  #0.0071 is the griper half length
                                    self.model.table_top_offset[2] - obj_bottom_offset[2]])
                obj_angle = np.pi/2.
                obj_quat = np.array([np.cos(obj_angle/2.), 0., 0., np.sin(obj_angle/2.)])

            # Concatenate to get the object qpos
            obj_qpos = np.concatenate([obj_pos, obj_quat])

            self.sim.data.qpos[self._ref_object_pos_low:self._ref_object_pos_high] = obj_qpos
            self.sim.forward()

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

            # max joint angles reward
            joint_limits = self._joint_ranges
            current_joint_pos = self._joint_positions

            hitting_limits_reward = - int(any([(x < joint_limits[i, 0] + 0.05 or x > joint_limits[i, 1] - 0.05) for i, x in
                                              enumerate(current_joint_pos)]))

            reward += hitting_limits_reward

            # reaching reward
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos[:2] - object_pos[:2])
            reaching_reward = -0.1 * dist
            reward += reaching_reward

            # Success Reward
            success = self._check_success()
            if (success):
                reward += 0.1

            # goal distance reward
            goal_pos = self.sim.data.site_xpos[self.goal_site_id]

            dist = np.linalg.norm(goal_pos[:2] - object_pos[:2])
            goal_distance_reward = -dist
            reward += goal_distance_reward

            unstable = reward < -2.5

            # Return all three types of rewards
            reward = {"reward": reward, "reaching_distance": -10 * reaching_reward,
                      "goal_distance": - goal_distance_reward,
                      "hitting_limits_reward": hitting_limits_reward,
                      "unstable":unstable}

        return reward

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        object_pos = self.sim.data.body_xpos[self.object_body_id][:2]
        goal_pos = self.sim.data.site_xpos[self.goal_site_id][:2]

        dist = np.linalg.norm(goal_pos - object_pos)
        goal_horizontal_radius = self.model.mujoco_goal.get_horizontal_radius()

        # object centre is within the goal radius
        return dist < goal_horizontal_radius

    def _pre_action(self, action):
        """ Takes the action, randomised the control timestep, and adds some additional random noise to the action."""
        # Change control timestep to simulate various random time delays
        timestep_parameter = self.dynamics_parameters['timestep_parameter']
        self.control_timestep = self.init_control_timestep + self.np_random.exponential(scale=timestep_parameter)

        # Add action noise to simulate unmodelled effects
        additive_noise = self.dynamics_parameters['action_additive_noise'] * self.np_random.uniform(-1, 1, action.shape)
        additive_systematic_noise = self.dynamics_parameters['action_systematic_noise']
        multiplicative_noise = 1.0 + (
                    self.dynamics_parameters['action_multiplicative_noise'] * self.np_random.uniform(-1, 1,
                                                                                                     action.shape))

        action = action * (1.0 + additive_noise + additive_systematic_noise) * multiplicative_noise

        super()._pre_action(action)

        # Adding forces

        # Addding forces to the joint
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes
        ] += self.dynamics_parameters['joint_forces'] * self.np_random.uniform(-1, 1, 7)

        # Adding force proportional to acceleration
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes
        ] += self.dynamics_parameters['acceleration_forces'] * self.sim.data.qacc[
            self._ref_joint_vel_indexes
        ]

        self.sim.data.xfrc_applied[
            self._ref_gripper_body_indx
        ] = self.dynamics_parameters['eef_forces'] * self.np_random.uniform(-1, 1, 6)

        # Adding forces to the object
        obj_qvel_low_idx , obj_qvel_high_idx = self.sim.model.get_joint_qvel_addr('rectangle')
        self.sim.data.qfrc_applied[
            obj_qvel_low_idx: obj_qvel_high_idx
        ] += self.dynamics_parameters['obj_forces'] * self.np_random.uniform(-1, 1, 6)

    def _post_action(self, action):
        """
        Add dense reward subcomponents to info, and checks for success of the task.
        """
        reward, done, info = super()._post_action(action)

        if self.reward_shaping:
            info = reward
            reward = reward["reward"]

        if(info["unstable"]):
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

        push_obj_name = self.model.object_names[self.model.push_object_idx]
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
            # Extract position and velocity of the eef
            eef_pos_in_world = self.sim.data.get_body_xpos("right_hand")
            eef_xvelp_in_world = self.sim.data.get_body_xvelp("right_hand")

            # Apply time delays
            eef_pos_in_world = self._apply_time_delay(eef_pos_in_world, self.eef_pos_queue)
            eef_xvelp_in_world = self._apply_time_delay(eef_xvelp_in_world, self.eef_vel_queue)

            # Add random noise
            position_noise = self.dynamics_parameters['eef_obs_position_noise']
            velocity_noise = self.dynamics_parameters['eef_obs_velocity_noise']

            eef_pos_in_world = eef_pos_in_world + self.np_random.normal(loc=0., scale=position_noise)
            eef_xvelp_in_world = eef_xvelp_in_world + self.np_random.normal(loc=0., scale=velocity_noise)

            # Get the position, velocity, rotation  and rotational velocity of the object in the world frame
            object_pos_in_world = self.sim.data.body_xpos[self.object_body_id]
            object_xvelp_in_world = self.sim.data.get_body_xvelp(push_obj_name)
            object_rot_in_world = self.sim.data.get_body_xmat(self.push_obj_name)

            # Apply time delays
            object_pos_in_world = self._apply_time_delay(object_pos_in_world, self.obj_pos_queue)
            object_xvelp_in_world = self._apply_time_delay(object_xvelp_in_world, self.obj_vel_queue)
            object_rot_in_world = self._apply_time_delay(object_rot_in_world, self.obj_angle_queue)

            # Get the z-angle with respect to the reference position and do sin-cosine encoding
            world_rotation_in_reference = np.array([[0., 1., 0., ], [-1., 0., 0., ], [0., 0., 1., ]])
            object_rotation_in_ref = world_rotation_in_reference.dot(object_rot_in_world)
            object_euler_in_ref = T.mat2euler(object_rotation_in_ref)
            z_angle = object_euler_in_ref[2]

            # Add random noise
            position_noise = self.dynamics_parameters['obj_obs_position_noise']
            velocity_noise = self.dynamics_parameters['obj_obs_velocity_noise']
            angle_noise = self.dynamics_parameters['obj_angle_noise']

            object_pos_in_world = object_pos_in_world + self.np_random.normal(loc=0., scale=position_noise)
            object_xvelp_in_world = object_xvelp_in_world + self.np_random.normal(loc=0., scale=velocity_noise)
            z_angle = z_angle + self.np_random.normal(loc=0., scale=angle_noise)


            # construct vectors for policy observation
            sine_cosine = np.array([np.sin(8*z_angle), np.cos(8*z_angle)])


            # Get the goal position in the world
            goal_site_pos_in_world = np.array(self.sim.data.site_xpos[self.goal_site_id])

            # Get the eef to object and object to goal vectors in EEF frame
            eef_to_object_in_world = object_pos_in_world - eef_pos_in_world
            eef_to_object_in_eef = self.world_rot_in_eef.dot(eef_to_object_in_world)

            object_to_goal_in_world = goal_site_pos_in_world - object_pos_in_world
            object_to_goal_in_eef = self.world_rot_in_eef.dot(object_to_goal_in_world)

            # Get the object's and the eef's velocities in EED frame
            object_xvelp_in_eef = self.world_rot_in_eef.dot(object_xvelp_in_world)
            eef_xvelp_in_eef = self.world_rot_in_eef.dot(eef_xvelp_in_world)


            # Record observations into a dictionary
            di['goal_pos_in_world'] = goal_site_pos_in_world
            di['eef_pos_in_world'] = eef_pos_in_world
            di['eef_vel_in_world'] = eef_xvelp_in_world
            di['object_pos_in_world'] = object_pos_in_world
            di['object_vel_in_world'] = object_xvelp_in_world
            di["z_angle"] = np.array([z_angle])

            di["task-state"] = np.concatenate(
                [eef_to_object_in_eef[:2],object_to_goal_in_eef[:2],
                 eef_xvelp_in_eef[:2], object_xvelp_in_eef[:2],
                 sine_cosine]
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
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                    (self.sim.model.geom_id2name(contact.geom1) in self.gripper.contact_geoms()
                     and contact.geom2 == self.sim.model.geom_name2id(object))

                    or (self.sim.model.geom_id2name(contact.geom2) in self.gripper.contact_geoms()
                        and contact.geom1 == self.sim.model.geom_name2id(object))
            ):
                collision = True
                break
        return collision



    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to object
        if self.gripper_visualization:
            # get distance to object
            object_site_id = self.sim.model.site_name2id(self.model.object_names[self.model.push_object_idx])
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





# 'joint_dampings':  np.array([[1./56.3,1./2.0,1./2.4,1./3.0,1./2.2,1./3.0,1./23.8],
            #                              [56.3,2.0,2.4,3.0,2.2,3.0,23.8,]])


# parameter_ranges['kps'] = np.array([[1./6.9,1./4.7,1./3.6,1/2.2,1./1.1,1./2.0,1/5.6],
#                                     [6.9,4.6,2.2,3.6,1.1,2.0,5.6]])
# parameter_ranges['kis'] =  np.array([[1./1.03,1./1.45,1./1.75,1./12.5,1./1.6,1./191.8,1./1.4],
#                                      [1.03,1.45,1.75,12.5,1.6,191.8,1.35]])
# parameter_ranges['kds'] =  np.array([[1./5.7,1./81.2,1./4.6,1./21.4,1./3.,1./2.1,1./3.],
#                                      [5.7,81.2,24.6,21.4,3.,2.1,3.,]])