import numpy as np

import gym
from gym.spaces import Discrete, Box, Tuple

from game_client import RPSClient
from adapter import RPSAdapter, my_id
from adapter import FEATURES_PER_PLANET, MAX_PLANETS
from adapter import FEATURES_PER_FLEET


class SpaceEnv(gym.Env):

    observation_space = Box(
        low=-1,
        high=np.inf,
        shape=(FEATURES_PER_PLANET * MAX_PLANETS,),
        dtype=np.float32)

    action_space = Tuple([
        Discrete(MAX_PLANETS),  # from
        Discrete(MAX_PLANETS),  # to
        Box(low=0, high=1, shape=(3,), dtype=np.float32),  # ships(x3)
    ])

    def __init__(self):
        self._adapter = RPSAdapter()
        self._client = None
        self._last_planets_count = 1

    def _lazy_init(self):
        if self._client:
            return
        self._client = RPSClient()

    def reset(self):
        self._lazy_init()

        self._last_planets_count = 1
        self._adapter.reset()
        self._client.reset()
        state, _, _ = self._fetch()
        return state

    def step(self, action):
        cmd = self._adapter.to_game_action(action)
        self._client.action(cmd)
        state, reward, done = self._fetch()
        return state, reward, done, {}

    def _fetch(self):
        game_state = self._client.game_state()
        state = self._adapter.parse_game_state(game_state)

        done = game_state["game_over"]

        cur_planets_count = count_planets(game_state)
        reward = cur_planets_count - self._last_planets_count
        self._last_planets_count = cur_planets_count

        return state, reward, done


def count_planets(game_state) -> int:
    player_id = my_id(game_state)
    planets = [
        planet for planet in game_state["planets"] if planet["owner_id"] == player_id]
    return len(planets)


def create_env(env_config):
    return SpaceEnv()
