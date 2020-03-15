from mujoco_py import MujocoException
import numpy as np
from robosuite_extra.utils import  transform_utils as T


def log_trajectory_point (env, obs, action,i, mujoco_elapsed, info,  logger, data_grabber):

    if(env._name == 'SawyerReach'):
        base_pos_in_world = env.sim.data.get_body_xpos("base")
        base_rot_in_world = env.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)
        base_rot_in_eef = env.init_right_hand_orn.T

        eef_pos_in_base, eef_vel_in_base, goal_pos_in_base = data_grabber(info, world_pose_in_base)
        action_in_base = base_rot_in_eef.dot(action)

        logger.log(i, mujoco_elapsed,
                   action_in_base[0], action_in_base[1], action_in_base[2],
                   eef_pos_in_base[0], eef_pos_in_base[1], eef_pos_in_base[2],
                   eef_vel_in_base[0], eef_vel_in_base[1], eef_vel_in_base[2],
                   goal_pos_in_base[0], goal_pos_in_base[1], goal_pos_in_base[2],
                   obs[0], obs[1], obs[2],
                   obs[3], obs[4], obs[5],
                   )

    elif(env._name == 'SawyerPush'):
        base_pos_in_world = env.sim.data.get_body_xpos("base")
        base_rot_in_world = env.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)
        base_rot_in_eef = env.init_right_hand_orn.T


        goal_pos_in_base, eef_pos_in_base, eef_vel_in_base, \
        object_pos_in_base, object_vel_in_base, z_angle, = data_grabber(info, world_pose_in_base)

        action_3d = np.concatenate([action, [0.0]])
        action_3d_in_base = base_rot_in_eef.dot(action_3d)

        logger.log(i, mujoco_elapsed,
                   action_3d_in_base[0], action_3d_in_base[1],
                   goal_pos_in_base[0], goal_pos_in_base[1], goal_pos_in_base[2],
                   eef_pos_in_base[0], eef_pos_in_base[1], eef_pos_in_base[2],
                   eef_vel_in_base[0], eef_vel_in_base[1], eef_vel_in_base[2],
                   object_pos_in_base[0], object_pos_in_base[1], object_pos_in_base[2],
                   object_vel_in_base[0], object_vel_in_base[1], object_vel_in_base[2],
                   z_angle[0],
                   obs[0], obs[1], obs[2],
                   obs[3], obs[4], obs[5],
                   obs[6], obs[7], obs[8],
                   obs[9],
                   )

    elif(env._name == ' SawyerSlide'):

        logger.log(i, mujoco_elapsed,
                   action[0], action[1],
                   obs[0], obs[1], obs[2],
                   obs[3], obs[4], obs[5], obs[6],
                   obs[7], obs[8], obs[9],
                   obs[10], obs[11],
                   )



def EPIpolicy_rollout(env,  epi_policy,s, mujoco_start_time , logger= None,data_grabber= None, max_steps=30, params=None):
    """
    Roll one trajectory with max_steps
    return: 
    traj: shape of (max_steps, state_dim+action_dim+reward_dim)
    s_: next observation
    env.get_state(): next underlying state
    """
    if params is not None:
        env.set_dynamics_parameters(params)
    # env.renderer_on()
    for epi_iter in range(10): # max iteration for getting a single rollout
        #TODO: Check this change: the environment has just been reset when doning the epi, we pass the resulting state in the arguments
        #s = env.reset()
        traj=[]
        for _ in range(max_steps):
            a = epi_policy.get_action(s)
            a = np.clip(a, -epi_policy.action_range, epi_policy.action_range)
            # env.render()
            try:
                mujoco_elapsed = env.sim.data.time - mujoco_start_time
                s_, r, done, info = env.step(a)
            except MujocoException:
                print('EPI Rollout: MujocoException')
                break

            if(logger is not None):
                log_trajectory_point(env, s_,a,epi_iter, mujoco_elapsed ,info,logger,data_grabber)

            s_a_r = np.concatenate((s,a, [r]))  # state, action, reward
            traj.append(s_a_r)
            s=s_
        if len(traj) == max_steps:
            break        

    if len(traj)<max_steps:
        print('EPI rollout length smaller than expected!')
        
    return traj, [s_, env.get_state()]