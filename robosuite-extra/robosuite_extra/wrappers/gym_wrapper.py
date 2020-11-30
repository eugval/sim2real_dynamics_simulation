'''
Taken and  modified from the original robosuite repository (version 0.1.0)
Our fork with version 0.1.0 : https://github.com/eugval/robosuite
Official Robosuite Repository : https://github.com/ARISE-Initiative/robosuite

'''

import numpy as np
from gym import spaces
from robosuite.wrappers import Wrapper

from gym import Wrapper as OpenAIGymWrapper



class FlattenWrapper(OpenAIGymWrapper):

    def __init__(self, env, keys=None, add_info=False):

        super().__init__(env)

        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "task-state", "target_pos",]
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        self.add_info = add_info

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)

        if(self.add_info):
            info.update(ob_dict)

        return self._flatten_obs(ob_dict), reward, done, info


    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        return self.env.__getattr__(attr)





class GymWrapper(Wrapper):
    env = None

    # Set this in SOME subclasses
    metadata = {'render.modes': ['human', 'rgb_a']}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, env):
        """
        Initializes the Gym wrapper.
        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        self.env = env
        # set up observation space
        high = np.inf
        low = -high

        obs_spec = env.observation_spec()

        space_spec = {}

        for k,v in obs_spec.items():
            space_spec[k]=spaces.Box(low=low,high=high, shape=v)


        self.observation_space = spaces.Dict(space_spec)

        # setup action space
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        self.reward_range = self.env.reward_range

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        ob_dict = self.env.reset()
        return ob_dict

    def render(self, mode):
        self.env.render()


    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

    @property
    def action_spec(self):
        return self.env.action_spec

    @property
    def unwrapped(self):
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False
