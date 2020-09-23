import numpy as np
from robosuite_extra.env_base import make
from robosuite_extra.wrappers import EEFXVelocityControl, GymWrapper, FlattenWrapper
from sim2real_policies.final_policy_testing.network_loading import load, load_model
import os
from sim2real_policies.final_policy_testing.epi_utils import EPIpolicy_rollout
import copy

REACH_GOALS= [ np.array([0.,0.]),np.array([-4.465e-2,5.85e-2]),np.array([8.37e-2,-5.78e-2])]
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def grab_data(info, world_pose_in_base):
    # Grab the data
    eef_pos_in_world = info['eef_pos_in_world']
    eef_vel_in_world = info['eef_vel_in_world']
    goal_pos_in_world = info['goal_pos_in_world']

    eef_pos_in_world = np.concatenate([eef_pos_in_world, [1.0]])
    goal_pos_in_world = np.concatenate([goal_pos_in_world, [1.0]])

    eef_pos_in_base = world_pose_in_base.dot(eef_pos_in_world)
    eef_pos_in_base = eef_pos_in_base / eef_pos_in_base[3]
    eef_vel_in_base = world_pose_in_base[:3, :3].dot(eef_vel_in_world)

    goal_pos_in_base = world_pose_in_base.dot(goal_pos_in_world)
    goal_pos_in_base = goal_pos_in_base /  goal_pos_in_base[3]


    return eef_pos_in_base, eef_vel_in_base, goal_pos_in_base


def find_zvalues(save_path):
    render = False

    # Parameters
    horizon = 50
    total_repeats =50


    env_name = 'SawyerReach'

    ### Prepare Environment #####
    env = make(
        'SawyerReach',
        gripper_type="PushingGripper",
        parameters_to_randomise=[],
        randomise_initial_conditions=True,
        table_full_size=(0.8, 1.6, 0.719),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        use_indicator_object=False,
        has_renderer=render,
        has_offscreen_renderer=False,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=100,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        pid=True,
        success_radius=0.01
    )

    env = FlattenWrapper(GymWrapper(EEFXVelocityControl(env, dof=3,max_action = 0.1),),keys='task-state',add_info= True)
    env.reset()

    z_values = []

    for param_iteration in range(3):
        state_dim = 6
        action_dim = 3
        z_values.append([])

        ### setting up the env with different but fixed parameters###
        if(param_iteration == 1):
            parameter_ranges = env.parameter_sampling_ranges
            max_parameters = dict([(k,v[-1]*env.factors_for_param_randomisation[k]) for k,v in parameter_ranges.items()])
            env.set_dynamics_parameters(max_parameters)
        elif(param_iteration == 2):
            parameter_ranges = env.parameter_sampling_ranges
            min_parameters = dict([(k,v[0]*env.factors_for_param_randomisation[k]) for k, v in parameter_ranges.items()])
            env.set_dynamics_parameters(min_parameters)

        env.reset()

        if (render):
            env.viewer.set_camera(camera_id=0)
            env.render()

        ###############

        ################# SETTING UP THE POLICY #################
        method = 'EPI'
        alg_name = 'epi_td3'
        embed_dim = 10
        traj_l = 10
        NO_RESET = True
        embed_input_dim = traj_l*(state_dim+action_dim)
        ori_state_dim = state_dim
        state_dim += embed_dim

        # choose which randomisation is applied
        number_random_params = 14
        folder_path = '../../../../sawyer/src/sim2real_dynamics_sawyer/assets/rl/'+method +'/' + alg_name + '/model/'
        path = folder_path + env_name + str(
            number_random_params) + '_' + alg_name

        embed_model = load_model(model_name='embedding', path=path, input_dim = embed_input_dim, output_dim = embed_dim )
        embed_model.cuda()
        epi_policy_path = folder_path + env_name + str(number_random_params) + '_' + 'epi_ppo_epi_policy'
        epi_policy = load(path=epi_policy_path, alg='ppo', state_dim=ori_state_dim, action_dim=action_dim )

        policy = load(path=path, alg=alg_name, state_dim=state_dim,
                        action_dim=action_dim)
        #########################################################

        for repeat in range(total_repeats):
            #Reset environment
            obs = env.reset()
            i=0

            mujoco_start_time = env.sim.data.time
            if NO_RESET:
                i = traj_l - 1
                traj, [last_obs, last_state] = EPIpolicy_rollout(env, epi_policy, obs,
                                                                 mujoco_start_time=mujoco_start_time,
                                                                 logger=None, data_grabber=None,
                                                                 max_steps=traj_l,
                                                                 params=None)  # only one traj; pass in params to ensure it's not changed
                state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
                embedding = embed_model(state_action_in_traj.reshape(-1))
                embedding = embedding.detach().cpu().numpy()

                obs = last_obs  # last observation
                env.set_state(last_state)  # last underlying state
            else:

                traj, [last_obs, last_state] = EPIpolicy_rollout(env, epi_policy, obs,
                                                                 mujoco_start_time=mujoco_start_time,
                                                                 logger=None, data_grabber=None,
                                                                 max_steps=traj_l,
                                                                 params=None)  # only one traj; pass in params to ensure it's not changed
                state_action_in_traj = np.array(traj)[:, :-1]  # remove the rewards
                embedding = embed_model(state_action_in_traj.reshape(-1))
                embedding = embedding.detach().cpu().numpy()


                params = env.get_dynamics_parameters()
                env.randomisation_off()
                env.set_dynamics_parameters(params)  # same as the rollout env
                obs = env.reset()
                env.randomisation_on()

            z_values[param_iteration].append(embedding)

            # z is embedding of initial trajectory for each episode, so no need to run task-policy rollout below

            while (True):

                ############# CHOOSING THE ACTION ##############
                obs = np.concatenate((obs, embedding))

                action = policy.get_action(obs)

                ################################################

                next_obs, reward, done, info = env.step(action)

                obs = next_obs

                if(render):
                    env.render()

                i += 1
                if (i >= horizon):
                    break

    z_values = np.array(z_values)
    env.close()

    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
    pickle.dump(z_values, open(os.path.join(save_path, 'z_values_array.pckl'), 'wb'))

    return z_values




def plot_tsne(z_values,save_path):

    tsne_calculator = TSNE(n_iter = 3000, n_iter_without_progress= 600,  perplexity =5.)

    fig = plt.figure()

    colors = plt.get_cmap('rainbow')(np.linspace(0.2, 1, 3))

    for param_set_idx in range(z_values.shape[0]):
        zs = copy.deepcopy(z_values[param_set_idx,:,:])
        zs_2d = tsne_calculator.fit_transform(zs)

        zs_2d_x = [zs_2d[i][0] for i in range(zs_2d.shape[0])]
        zs_2d_y = [zs_2d[i][1] for i in range(zs_2d.shape[0])]

        plt.scatter(zs_2d_x, zs_2d_y, c = [colors[param_set_idx]]*len(zs_2d_y), label='params set {}'.format(param_set_idx+1))

    plt.legend()
    plt.axis('off')
    plt.savefig(save_path+'tsne_plot.pdf')
    plt.savefig(save_path+'tsne_plot.svg')

    plt.show()
    plt.close()




if __name__ == "__main__":


    save_path = '../../../../data/reaching/epi_tsne_results/'

    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), save_path)
    #z_values = find_zvalues(save_path)
    #


    z_values = pickle.load(open(os.path.join(save_path, 'z_values_array.pckl'), 'rb'))
    plot_tsne(copy.deepcopy(z_values),save_path)


