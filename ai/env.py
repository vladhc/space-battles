import numpy as np

import gym
from gym.spaces import Discrete, Box, Tuple

from game_client import RPSClient
from adapter import RPSAdapter
from adapter import FEATURES_PER_PLANET, MAX_PLANETS
from adapter import FEATURES_PER_FLEET, MAX_FLEETS_PER_PLANET


class SpaceEnv(gym.Env):

    observation_space = Box(
        low=0,
        high=5,
        shape=(FEATURES_PER_PLANET * MAX_PLANETS,),
        dtype=np.int)
    action_space = Discrete(1)

    def __init__(self):
        self._adapter = RPSAdapter()
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
        action = (0, 0, 0, 0, 0, 1)
        cmd = self._adapter.to_game_action(action)
        self._client.action(cmd)
        state, reward, done = self._fetch()
        return state, reward, done, {}

    def _fetch(self):
        game_state = self._client.game_state()
        state = self._adapter.parse_game_state(game_state)
        done = game_state["game_over"]
        return state, 0, done


def create_env(env_config):
    return SpaceEnv()
