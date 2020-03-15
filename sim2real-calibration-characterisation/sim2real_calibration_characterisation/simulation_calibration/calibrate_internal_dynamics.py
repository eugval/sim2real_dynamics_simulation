import numpy as np
from robosuite_extra.env_base import make
from robosuite_extra.wrappers import EEFXVelocityControl
import math
import os
import time
from scipy.optimize import differential_evolution

import pickle
from mujoco_py import MujocoException

start = time.time()


########## Script parameters ###########
_REAL_WORLD_MATCHING = False
LOG_PATH = './param_calibration.pckl' # Where to save the optimised values of the parameters

if (_REAL_WORLD_MATCHING == True):
    path = ""
    real_trajectories = pickle.load(open(path, 'rb'), encoding='latin1')

######################################



_POLICIES = [{'policy_type': 'joint', 'period': 0.4, 'amplitude': 0.2, 'line_policy': None, 'steps': 50},
             {'policy_type': 'joint', 'period': 0.3, 'amplitude': 0.2, 'line_policy': None, 'steps': 50},
             {'policy_type': 'joint', 'period': 0.5, 'amplitude': 0.2, 'line_policy': None, 'steps': 50},
             {'policy_type': 'eef_circle', 'period': 0.2, 'amplitude': 0.05, 'line_policy': None, 'steps': 50},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None, 'line_policy': np.array([0.0, 0.0, 0.05]),
              'steps': 36},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None, 'line_policy': np.array([0.05, 0.05, 0.05]),
              'steps': 36},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None,
              'line_policy': np.array([-0.05, 0.05, 0.05]), 'steps': 36},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None, 'line_policy': np.array([0.05, 0.0, 0.0]),
              'steps': 36},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None, 'line_policy': np.array([0.0, 0.05, 0.0]),
              'steps': 36},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None, 'line_policy': np.array([0.0, -0.05, 0.0]),
              'steps': 24},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None, 'line_policy': np.array([-0.05, 0.0, 0.0]),
              'steps': 36},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None, 'line_policy': np.array([0.05, 0.05, 0.0]),
              'steps': 36},
             {'policy_type': 'eef_line', 'period': None, 'amplitude': None,
              'line_policy': np.array([-0.05, -0.05, 0.0]), 'steps': 18},
             ]


class PolicyGenerator(object):
    ''' Uses the predefined open loop policy dictionary _POLICIES to give actions for the simulator at each timestep. '''
    def __init__(self, policy_type='joint', line_policy=np.array([0.0, 0.0, 0.0]), period_factor=0.15,
                 amplitude_factor=0.1):
        # Sin-cos control
        assert policy_type == 'joint' or policy_type == 'eef_circle' or policy_type == 'eef_line'

        self.policy_type = policy_type

        self.line_policy = line_policy

        self.period_factor = period_factor
        self.amplitude_factor = amplitude_factor
        self._right_joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

    def set_policy_type(self, policy_type):
        assert policy_type == 'joint' or policy_type == 'eef_circle' or policy_type == "eef_line"

        self.policy_type = policy_type

    def set_period_amplitude(self, period, amplitude):
        if (period is not None):
            self.period_factor = period
        if (amplitude is not None):
            self.amplitude_factor = amplitude

    def set_line_policy(self, line_policy):
        if (line_policy is not None):
            self.line_policy = line_policy

    def cos_wave(self, elapsed):
        w = self.period_factor * elapsed
        return self.amplitude_factor * math.cos(w * 2 * math.pi)

    def sin_wave(self, elapsed):
        w = self.period_factor * elapsed
        return self.amplitude_factor * math.sin(w * 2 * math.pi)

    def get_control(self, time, theshold_time=-1.):
        if (self.policy_type == 'joint'):
            return np.array([self.sin_wave(time) for _ in range(7)])

        elif (self.policy_type == 'eef_circle'):
            return np.array([self.sin_wave(time), self.cos_wave(time), 0.0])

        elif (self.policy_type == 'eef_line'):

            if (time < theshold_time):
                return self.line_policy
            else:
                return np.array([0., 0., 0.])

def function_to_optimise(arg_array):
    ''' The cost function for optimising the simulator parameters. It can run in two settings, depending on
    the REAL_WORLD_MATCHING flag. If True, this functhion will compare the generated trajectories with
    real world trajectories obtained with the same policies. If False, this function will assess how well
    the robot responds to the control command.'''

    # Parameters to optimise
    kp0, kp1, kp2, kp3, kp4, kp5, kp6, \
    ki0, ki1, ki2, ki3, ki4, ki5, ki6, \
    kd0, kd1, kd2, kd3, kd4, kd5, kd6, \
    damping0, damping1, damping2, damping3, damping4, damping5, damping6,\
    armature0, armature1, armature2, armature3, armature4, armature5, armature6, \
    friction0, friction1, friction2, friction3, friction4, friction5, friction6= arg_array

    if(not _REAL_WORLD_MATCHING):
        command_trajectories = []

    # Create Environment
    env = make(
        'SawyerReach',
        gripper_type="PushingGripper",
        parameters_to_randomise=[],
        randomise_initial_conditions=False,
        table_full_size=(0.8, 1.6, 0.719),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_collision_mesh=False,
        render_visual_mesh=False,
        control_freq=10,
        horizon=200000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        pid=True,
        success_radius=0.01
    )

    env.normalised_actions = False

    # Load params in environment
    params = {
        'kps': np.array([kp0, kp1, kp2, kp3, kp4, kp5, kp6, ]),
        'kis': np.array([ki0, ki1, ki2, ki3, ki4, ki5, ki6, ]),
        'kds': np.array([kd0, kd1, kd2, kd3, kd4, kd5, kd6, ]),
        'joint_dampings': np.array([damping0, damping1, damping2, damping3, damping4, damping5, damping6]),
        'armatures':np.array([armature0,armature1,armature2,armature3,armature4,armature5,armature6]),
        'joint_frictions':np.array([friction0, friction1,friction2,friction3,friction4,friction5,friction6,]),
    }

    env.set_dynamics_parameters(params)

    policy = PolicyGenerator()

    trajectories_gathered = []

    # For each of the open loop policies
    for policy_id, policy_params in enumerate(_POLICIES):
        trajectories_gathered.append([])

        if (not _REAL_WORLD_MATCHING):
            command_trajectories.append([])

        # Wrap the environment correctly if the policy is in end-effector space (environment out of the loop so only wrap once).
        if (policy_id == 3):
            env = EEFXVelocityControl(env, dof=3, normalised_actions=False)

        # Set the policy generator to the current policy
        policy.set_policy_type(policy_params['policy_type'])
        policy.set_period_amplitude(policy_params['period'], policy_params['amplitude'])
        policy.set_line_policy(policy_params['line_policy'])

        steps = policy_params['steps']
        env.reset()
        mujoco_start_time = env.sim.data.time
        #Run the trajectory
        for i in range(steps):
            mujoco_elapsed = env.sim.data.time - mujoco_start_time

            # get the observation
            eef_pos_in_base = env._right_hand_pos
            eef_vel_in_base = env._right_hand_vel
            obs_joint_pos = env._joint_positions
            obs_joint_vels = env._joint_velocities

            # act
            action = policy.get_control(mujoco_elapsed, theshold_time=steps * (5. / 60.))
            _, _, _, info = env.step(action)

            # Record
            if (_REAL_WORLD_MATCHING):
                trajectories_gathered[-1].append([eef_pos_in_base, eef_vel_in_base,
                                                  obs_joint_pos,
                                                  obs_joint_vels])
            else:
                if (policy_id >= 3):
                    commanded_velocity = info['joint_velocities']
                else:
                    commanded_velocity = action
                command_trajectories[-1].append(commanded_velocity)
                trajectories_gathered[-1].append(obs_joint_vels)

    # Find the cost
    loss = 0.0

    if (_REAL_WORLD_MATCHING):
        for j, sim_traj in enumerate(trajectories_gathered):
            real_traj = real_trajectories[j]
            for k, sim_datapoint in enumerate(sim_traj):
                real_datapoint = real_traj[k]
                for l, sim_item in enumerate(sim_datapoint):
                    real_item = real_datapoint[l]
                    loss += np.linalg.norm(real_item - sim_item)
    else:
        for j, sim_traj in enumerate(trajectories_gathered):
            command_traj = command_trajectories[j]
            for k, sim_datapoint in enumerate(sim_traj):
                command_datapoint = command_traj[k]
                loss += np.linalg.norm(command_datapoint - sim_datapoint)

    return loss




def CEM_covariance_finding(log_path, mean):
    '''This function can be used to find the covariance of a normal sampling distribution for the parameters with the
    mean already found by the differential evolution procedure. This covariance will be such that the the cost function will
    be minimised'''


    covariance = np.eye(28) * 0.5 * mean

    iterations = 20

    number_of_samples = 1000
    number_of_accepted_samples = 150

    for iteration in range(iterations):
        # sample parameters from gaussian
        param_samples = np.clip(np.random.multivariate_normal(mean, covariance, size=number_of_samples),
                                a_min=0.0000001, a_max=200.)

        function_values = np.zeros((number_of_samples,))

        for sample in range(number_of_samples):
            # evaluate function on each samples
            if (sample % 100 == 0):
                print('did {} \r'.format(sample))

            params = param_samples[sample, :]

            try:
                function_values[sample] = function_to_optimise(params)
            except MujocoException:
                print('Mujoco exception for params {}'.format(params))

        # order resulting samples following the function
        min_value_idxs = function_values.argsort(axis=0)[:number_of_accepted_samples]

        valid_param_samples = param_samples[min_value_idxs, :].T

        # fit new mean and new covariance to 20% top samples
        covariance = np.cov(valid_param_samples)

        print('Done {}%'.format((iteration + 1) * 10))
        print('current mean : {}'.format(mean))
        print('Current covariance : {}'.format(covariance))
        print('mean_function_value of best samples: {}'.format(np.mean(function_values[min_value_idxs])))
        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                log_path)
        pickle.dump(covariance, open(os.path.abspath(log_path), 'wb'))


    return covariance




def progress_bar_callback(xk, convergence):
    ''' Callback to the differencial evolution algorithm. Returns and saves the current best estimate of the parameters'''

    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            LOG_PATH)
    pickle.dump(xk, open(os.path.abspath(log_path), 'wb'))

    print('Current xk: {} , convergence: {}'.format(xk, convergence))



def main():
    ''' Use differencial evolution in order to find an estimate of the simulator internal dynamics parameters'''
    bounds = [(0., 100.), (0., 80.), (0., 60.), (0., 50.), (0., 40.), (0., 30.), (0., 20.),
              (0., 10.), (0., 8.), (0., 6.), (0., 5.), (0., 4.), (0., 3.), (0., 2.),
              (0., 5.), (0., 4.), (0., 3.), (0., 2.), (0., 1.), (0., 0.5), (0., 0.25),
              (0., 20.), (0., 15.), (0., 10.), (0., 8.), (0., 7.), (0., 6.), (0., 5.),
              (0.,5.),(0.,5.),(0.,5.),(0.,5.),(0.,5.),(0.,5.),(0.,5.),
              (0.,1.),(0.,1.),(0.,1.),(0.,1.),(0.,1.),(0.,1.),(0.,1.),
              ]
    result = differential_evolution(func=function_to_optimise,
                                    bounds=bounds, workers=-1, callback=progress_bar_callback, maxiter = 2000)


    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            LOG_PATH)
    pickle.dump(result, open(os.path.abspath(log_path), 'wb'))

    print(result)


if __name__ == '__main__':
    main()
