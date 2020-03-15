from robosuite_extra.env_base import make

from robosuite_extra.push_env.sawyer_push import SawyerPush
from robosuite_extra.reach_env.sawyer_reach import SawyerReach
from robosuite_extra.slide_env.sawyer_slide import SawyerSlide


def make_env(env_id, seed, rank, environment_arguments = {}, wrappers = [], wrapper_arguments = []):
    assert len(wrappers) == len(wrapper_arguments) , 'There needs to be one argument dict per wrapper'

    def _thunk():
        if env_id.startswith("gym"):
            raise NotImplementedError('Gym environments not implemented')
        elif env_id.startswith('robosuite'):
            _, robosuite_name = env_id.split('.')

            env = make(robosuite_name, **environment_arguments)
        else:
            raise NotImplementedError('Only  robosuite environments are compatible.')


        for i, wrapper in enumerate(wrappers):
            env = wrapper(env, **wrapper_arguments[i])

        env.seed(seed + rank)

        return env

    return _thunk

