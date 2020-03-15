def get_hyperparams():
    hyperparams_dict={
    'alg_name': 'td3',
    'action_range': 1.,
    'batch_size': 640,
    'explore_steps': 0,
    'update_itr': 1,  # iterative update
    'explore_noise_scale': 0.3, 
    'eval_noise_scale': 0.5,  # noisy evaluation trick
    'reward_scale': 1., # reward normalization
    'hidden_dim': 512,
    'noise_decay': 0.9999, # decaying exploration noise
    'policy_target_update_interval': 3, # delayed update
    'q_lr': 3e-4,
    'policy_lr': 3e-4,
    'replay_buffer_size': 1e6,
    'deterministic': True
    }
    print('Hyperparameters: ', hyperparams_dict)
    return hyperparams_dict
