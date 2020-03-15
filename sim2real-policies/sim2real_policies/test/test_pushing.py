import numpy as np
from robosuite_extra.env_base import make

# It is important to import the environment to use because it registers it for robosuite.make
from robosuite_extra.push_env.sawyer_push import SawyerPush
from rl_utils import load
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  # add parent path

from utils.choose_env import choose_env
import time
from robosuite_extra.wrappers import EEFXVelocityControl


#### Testing Parameter Generators : Need to choose one as the parameters to use in all
#    of the test environments across all methods need to be the same. These below do not take into account the case
#    of heavy randomisation


def testing_envionments_generator_linear(env, number_of_envs):
    default_parameters = env.get_default_dynamics_parameters()
    parameter_ranges = env._get_dynamics_parameter_ranges().copy()

    parameter_range_len = {}
    for k, v in parameter_ranges.items():
        parameter_range_len[k] = v[1] - v[0]

    uses_pid = bool(env.pid)

    for i in range(0, number_of_envs + 1):
        params = {}

        params['timestep_parameter'] = parameter_ranges['timestep_parameter'][0] + (i / number_of_envs) * \
                                       parameter_range_len['timestep_parameter']
        params['random_noise'] = parameter_ranges['random_noise'][0] + (i / number_of_envs) * \
                                 parameter_range_len['random_noise']
        params['obj_density'] = parameter_ranges['density'][0] + (i / number_of_envs) * \
                                parameter_range_len['density']
        params['obj_sliding_friction'] = parameter_ranges['sliding_friction'][0] + (i / number_of_envs) * \
                                         parameter_range_len['sliding_friction']
        params['obj_torsional_friction'] = parameter_ranges['torsional_friction'][0] + (i / number_of_envs) * \
                                           parameter_range_len['torsional_friction']

        gains_factor = parameter_ranges['gains'][0] + (i / number_of_envs) * \
                       parameter_range_len['gains']
        if (uses_pid):
            kps = default_parameters['kps'] * gains_factor
            kis = default_parameters['kis'] * gains_factor
            kds = default_parameters['kds'] * gains_factor

            params['kps'] = kps
            params['kis'] = kis
            params['kds'] = kds
        else:
            for j in range(7):
                gain_name = 'right_j{}_kv'.format(j)
                params[gain_name] = default_parameters[gain_name] * gains_factor

        yield params


_FIXED_RANDOM_NUMBERS = np.array([[0.2960349, 0.20024673, 0.3284088, 0.13947944, 0.23070504,
                                   0.52801767, 0.03546525, 0.67319874, 0.35803635, 0.65429358,
                                   0.31416152, 0.50058094, 0.10993917],
                                  [0.03328016, 0.01069134, 0.4048272, 0.59784418, 0.03386122,
                                   0.02654122, 0.29738415, 0.26494653, 0.56118631, 0.93254799,
                                   0.33608316, 0.97122181, 0.30445317],
                                  [0.34027698, 0.1824053, 0.0139382, 0.25615233, 0.20052826,
                                   0.13882747, 0.50738538, 0.75013958, 0.45730579, 0.26662392,
                                   0.4769783, 0.75964153, 0.55804381],
                                  [0.79246048, 0.2770366, 0.22946863, 0.79738624, 0.23766555,
                                   0.68886918, 0.76139074, 0.235938, 0.56851715, 0.98678031,
                                   0.12399767, 0.05368876, 0.60516519],
                                  [0.27259969, 0.05017477, 0.25374302, 0.38582487, 0.34148632,
                                   0.88377469, 0.93073029, 0.13556076, 0.80191293, 0.17453797,
                                   0.55516497, 0.9193243, 0.02106774],
                                  [0.50596479, 0.16848385, 0.32618285, 0.83633656, 0.47922389,
                                   0.82962022, 0.0448616, 0.58434176, 0.93891147, 0.72981041,
                                   0.94095669, 0.19927713, 0.05040289],
                                  [0.54156721, 0.58919206, 0.98744488, 0.35474934, 0.43873577,
                                   0.33923125, 0.37133492, 0.44613637, 0.01237529, 0.74946781,
                                   0.38701296, 0.16919404, 0.42697889],
                                  [0.5530095, 0.80434606, 0.15246865, 0.48090045, 0.7162523,
                                   0.68680996, 0.1245025, 0.24024631, 0.8825019, 0.75384631,
                                   0.98332077, 0.8009945, 0.44438732],
                                  [0.6135933, 0.70149082, 0.73271676, 0.8002691, 0.67998565,
                                   0.00797684, 0.54232372, 0.12034335, 0.21407514, 0.56841125,
                                   0.24849677, 0.99623495, 0.58355981],
                                  [0.9358823, 0.26157918, 0.093606, 0.92972687, 0.4052983,
                                   0.4316369, 0.75627628, 0.37273345, 0.51766127, 0.61298269,
                                   0.32708491, 0.50825398, 0.33808411]])


def testing_envionments_generator_random(env, number_of_envs):
    default_parameters = env.get_default_dynamics_parameters()
    parameter_ranges = env._get_dynamics_parameter_ranges().copy()

    parameter_range_len = {}
    for k, v in parameter_ranges.items():
        parameter_range_len[k] = v[1] - v[0]

    uses_pid = bool(env.pid)

    for i in range(0, number_of_envs):
        params = {}

        params['timestep_parameter'] = parameter_ranges['timestep_parameter'][0] + _FIXED_RANDOM_NUMBERS[i, 0] * \
                                       parameter_range_len['timestep_parameter']
        params['random_noise'] = parameter_ranges['random_noise'][0] + _FIXED_RANDOM_NUMBERS[i, 1] * \
                                 parameter_range_len['random_noise']
        params['obj_density'] = parameter_ranges['density'][0] + _FIXED_RANDOM_NUMBERS[i, 2] * \
                                parameter_range_len['density']
        params['obj_sliding_friction'] = parameter_ranges['sliding_friction'][0] + _FIXED_RANDOM_NUMBERS[i, 3] * \
                                         parameter_range_len['sliding_friction']
        params['obj_torsional_friction'] = parameter_ranges['torsional_friction'][0] + _FIXED_RANDOM_NUMBERS[i, 4] * \
                                           parameter_range_len['torsional_friction']

        if (uses_pid):
            kps = default_parameters['kps'] * parameter_ranges['gains'][0] + _FIXED_RANDOM_NUMBERS[i, 5] * \
                  parameter_range_len['gains']
            kis = default_parameters['kis'] * parameter_ranges['gains'][0] + _FIXED_RANDOM_NUMBERS[i, 6] * \
                  parameter_range_len['gains']
            kds = default_parameters['kds'] * parameter_ranges['gains'][0] + _FIXED_RANDOM_NUMBERS[i, 7] * \
                  parameter_range_len['gains']

            params['kps'] = kps
            params['kis'] = kis
            params['kds'] = kds
        else:
            for j in range(7):
                gain_name = 'right_j{}_kv'.format(j)
                params[gain_name] = default_parameters[gain_name] * \
                                    parameter_ranges['gains'][0] + _FIXED_RANDOM_NUMBERS[i, 5 + j] * \
                                    parameter_range_len['gains']

        yield params

######################################################

if __name__ == "__main__":

    policy = load(path = '../../../../data/sac/model/sac_v2_multiprocess_multi', alg='SAC')

    # env = make(
    #     'SawyerPush',
    #     gripper_type="PushingGripper",
    #     randomise_parameters=False,
    #     heavy_randomisation=True,
    #     randomise_starting_position=True,
    #     table_full_size=(0.8, 1.6, 0.719),
    #     table_friction=(1e-4, 5e-3, 1e-4),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     reward_shaping=True,
    #     placement_initializer=None,
    #     gripper_visualization=True,
    #     use_indicator_object=False,
    #     has_renderer=False,
    #     has_offscreen_renderer=False,
    #     render_collision_mesh=False,
    #     render_visual_mesh=True,
    #     control_freq=10,
    #     horizon=80,
    #     ignore_done=False,
    #     camera_name="frontview",
    #     camera_height=256,
    #     camera_width=256,
    #     camera_depth=False,
    #     pid=False,
    # )

    # env.seed(0)

    # # Extra wrapper to do end-effector velocity control
    # env = EEFXVelocityControl(env,
    #                           dof=2,
    #                           max_action=0.1,
    #                           normalised_actions=True)
    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env('SawyerPush')

    start_time = time.time()

    return_per_test_environment = []

    dynamics_params = testing_envionments_generator_random(env, 5)
    # do visualization
    print('Starting testing')
    for params in dynamics_params:
        env.set_dynamics_parameters(params)
        mean_return = 0.0

        print('with parameters {} ...'.format(params))
        number_of_episodes = 50
        for i in range(number_of_episodes):
            obs = env.reset()

            episode_return = 0.0

            start_time = time.time()
            done = False
            while not done:
                action = policy.get_action(obs)
                obs, reward, done, _ = env.step(action)
                episode_return += reward

                if (done):
                    mean_return += episode_return

        mean_return /= number_of_episodes
        print('the mean return is {}'.format(mean_return))
        return_per_test_environment.append(mean_return)

    print('total_average_return_over_test_environments : {}'.format(np.mean(return_per_test_environment)))
    print('it took {} minutes'.format((time.time() - start_time) / 60))
