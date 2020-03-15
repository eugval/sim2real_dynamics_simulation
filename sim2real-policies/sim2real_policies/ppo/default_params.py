def get_hyperparams():
    hyperparams_dict={
    'alg_name': 'ppo',
    'action_range': 1., # (-action_range, action_range)
    'batch_size': 128,  # update batchsize
    'gamma': 0.9,   # reward discount
    'random_seed': 2,  # random seed
    'actor_update_steps': 10,   # actor update iterations
    'critic_update_steps': 10,   # critic update iterations
    'eps': 1e-8, # numerical residual
    'actor_lr': 0.0001,  # learning rate for actor
    'critic_lr': 0.0002, # learning rate for critic
    'method': [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization
    }
    print('Hyperparameters: ', hyperparams_dict)
    return hyperparams_dict