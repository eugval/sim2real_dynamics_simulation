from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper
from sim2real_policies.utils.envs import make_env

push_no_randomisation = []

push_full_randomisation = ['eef_timedelay', 'obj_timedelay', 'timestep_parameter', 'pid_iteration_time',
                           'action_additive_noise', 'action_systematic_noise', 'action_multiplicative_noise',
                           'eef_obs_position_noise', 'eef_obs_velocity_noise',
                           'obj_obs_position_noise', 'obj_obs_velocity_noise', 'obj_angle_noise',
                           'obj_density', 'obj_sliding_friction', 'obj_torsional_friction', 'obj_size',
                           'link_masses', 'joint_dampings', 'armatures', 'joint_frictions',
                           'kps', 'kis', 'kds', ]

push_force_randomisation = ['joint_forces', 'obj_forces']

push_force_noise_randomisation = ['eef_timedelay', 'obj_timedelay',
                                  'eef_obs_position_noise', 'eef_obs_velocity_noise',
                                  'obj_obs_position_noise', 'obj_obs_velocity_noise', 'obj_angle_noise',
                                  'joint_forces', 'obj_forces']


reach_no_randomisation = []

reach_full_randomisation = ['action_additive_noise', 'action_systematic_noise', 'action_multiplicative_noise',
                            'eef_obs_position_noise', 'eef_obs_velocity_noise',
                            'timestep_parameter', 'pid_iteration_time',
                            'link_masses', 'joint_dampings', 'armatures', 'joint_frictions',
                            'kps', 'kis', 'kds']

reach_force_randomisation = ['joint_forces']

reach_force_noise_randomisation = ['eef_obs_position_noise', 'eef_obs_velocity_noise', 'joint_forces', ]


slide_no_randomisation = []

slide_full_randomisation = ['obj_timedelay', 'timestep_parameter', 'pid_iteration_time',
                            'action_additive_noise', 'action_multiplicative_noise', 'action_systematic_noise',
                            'joint_obs_position_noise', 'joint_obs_velocity_noise', 'obs_position_noise',
                            'obs_velocity_noise', 'obs_angle_noise',
                            'obj_density', 'obj_size', 'obj_sliding_friction', 'obj_torsional_friction',
                            'link_masses', 'joint_dampings', 'armatures', 'joint_frictions',
                            'kps', 'kis', 'kds']

slide_force_randomisation = ['joint_forces', 'obj_forces']

slide_force_noise_randomisation = ['joint_obs_position_noise', 'joint_obs_velocity_noise', 'obs_position_noise',
                                   'obs_velocity_noise', 'obs_angle_noise',
                                   'obj_timedelay',
                                   'joint_forces', 'obj_forces'
                                   ]


def choose_env(env_name, randomisation_configuration=None, env_params_dict=None):
    if env_name == 'SawyerPush':
        ## Set the environment parameters
        environment_params = {
            "gripper_type": "PushingGripper",
            "parameters_to_randomise": push_force_noise_randomisation,
            "randomise_initial_conditions": True,
            "table_full_size": (0.8, 1.6, 0.719),
            "table_friction": (1e-4, 5e-3, 1e-4),
            "use_camera_obs": False,
            "use_object_obs": True,
            "reward_shaping": True,
            "placement_initializer": None,
            "gripper_visualization": False,
            "use_indicator_object": False,
            "has_renderer": False,  # <--------
            "has_offscreen_renderer": False,
            "render_collision_mesh": False,
            "render_visual_mesh": True,
            "control_freq": 10,  # <--------
            "horizon": 80,  # <--------
            "ignore_done": False,
            "camera_name": "frontview",
            "camera_height": 256,
            "camera_width": 256,
            "camera_depth": False,
            "pid": True

        }

        environment_wrappers = [EEFXVelocityControl, GymWrapper, FlattenWrapper]
        environment_wrapper_arguments = [{'max_action': 0.1, 'dof': 2},
                                         {},
                                         {'keys': 'task-state'}]


    elif env_name == 'SawyerReach':
        ## Set the environment parameters
        environment_params = {
            "gripper_type": "PushingGripper",
            "parameters_to_randomise": reach_full_randomisation,
            "randomise_initial_conditions": True,
            "table_full_size": (0.8, 1.6, 0.719),
            "use_camera_obs": False,
            "use_object_obs": True,
            "reward_shaping": True,
            "use_indicator_object": False,
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "render_collision_mesh": False,
            "render_visual_mesh": True,
            "control_freq": 10.,
            "horizon": 50,
            "ignore_done": False,
            "camera_name": " frontview",
            "camera_height": 256,
            "camera_width": 256,
            "camera_depth": False,
            "pid": True,
            "success_radius": 0.01
        }

        environment_wrappers = [EEFXVelocityControl, GymWrapper, FlattenWrapper]
        environment_wrapper_arguments = [{'max_action': 0.1, 'dof': 3},
                                         {},
                                         {'keys': 'task-state'}]

    elif env_name == 'SawyerSlide':
        environment_params = {
            'gripper_type': "SlidePanelGripper",
            'parameters_to_randomise': slide_full_randomisation,
            'randomise_initial_conditions': True,
            'use_camera_obs': False,
            'use_object_obs': True,
            'reward_shaping': True,
            'use_indicator_object': False,
            'has_renderer': False,
            'has_offscreen_renderer': False,
            'render_collision_mesh': False,
            'render_visual_mesh': True,
            'control_freq': 10.,
            'horizon': 60,
            'ignore_done': False,
            'camera_name': "frontview",
            'camera_height': 256,
            'camera_width': 256,
            'camera_depth': False,
            'pid': True,
        }

        environment_wrappers = [ GymWrapper, FlattenWrapper]
        environment_wrapper_arguments = [{},
                                         {'keys': 'task-state'}]
    if randomisation_configuration is not None:
        environment_params['parameters_to_randomise'] = eval(randomisation_configuration)
    
    if env_params_dict is not None:
        for key, value in env_params_dict.items():
            try:
                environment_params[key] = value
            except Exception as error:
                print(error)


    env = make_env('robosuite.'+env_name, 1, 0, environment_params, environment_wrappers,
                    environment_wrapper_arguments)()

    print('Randomised Parameters: {}'.format(environment_params["parameters_to_randomise"]))
    return env, environment_params, environment_wrappers, environment_wrapper_arguments
