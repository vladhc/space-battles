import numpy as np

from ray.tune.registry import register_env
import gym
from gym.spaces import Discrete, Box

from game_client import RPSClient


FEATURES_PER_PLANET = 10
MAX_PLANETS = 20


class SpaceEnv(gym.Env):

    observation_space = Box(
        low=0,
        high=5,
        shape=(FEATURES_PER_PLANET * MAX_PLANETS),
        dtype=np.int8)
    action_space = Discrete(len(Command))

    def __init__(self):
        self._adapter = Adapter()
        self._client = None

    def _lazy_init(self):
        if self._client:
            return
        self._client = RPSClient()

    def reset(self):
        self._lazy_init()

        self._client.reset()
        state, _, _ = self._fetch()
        return state

    def step(self, action):
        cmd = self._adapter.action2cmd(action)
        self._client.send(cmd)
        state, reward, done = self._fetch()
        return state, reward, done, {}

    def _fetch(self):
        game_state = self._client.get_state()
        state, reward, done = self._adapter.parse_game_state(game_state)
        return state, reward, done


def create_env(env_config):
    return SpaceEnv()


register_env("SpaceEnv", create_env)
