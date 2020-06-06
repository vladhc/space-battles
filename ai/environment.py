import numpy as np
import json
import gym
from gym.spaces import Discrete, Box, Dict, Tuple

import ray

from game_client import RPSClient

from state import feed_dict

MAX_PLANETS = 21

NOOP = 'noop'

class SpaceEnv(gym.Env):
    action_space = Tuple([
        Discrete(MAX_PLANETS),  # from
        Discrete(MAX_PLANETS),  # to
        Box(low=0, high=1, shape=(3,), dtype=np.float32),  # ships(x3)
    ])

    def __init__(self, client_args):
        self._client_args = client_args
        self._client = None
        self._last_game_state = None
        self._last_planets_count = 1

    def _lazy_init(self):
        if self._client:
            return
        self._client = RPSClient(**self._client_args)

    def reset(self):
        self._lazy_init()

        self._client.reset()

    def step(self, action):
        planets = self._last_game_state['planets']

        cmd = to_game_action(planets, action)
        self._client.action(cmd)
        state, reward, done = self._fetch()
        return state, reward, done, {}

    def _fetch(self):
        game_state = self._client.game_state()
        self._last_game_state = game_state
        state = parse_game_state(game_state)

        done = game_state["game_over"]

        cur_planets_count = count_planets(game_state)

        reward = cur_planets_count - self._last_planets_count
        reward -= 0.01  # penalty to make them work faster
        self._last_planets_count = cur_planets_count

        return state, reward, done

def my_id(state):
    for player in state["players"]:
        if player["itsme"]:
            return player["id"]
    return None

def count_planets(game_state) -> int:
    player_id = my_id(game_state)
    planets = [
        planet for planet in game_state["planets"] if planet["owner_id"] == player_id]
    return len(planets)


def to_game_action(planets, action):
    planets = {planet["id"]: planet for planet in planets}

    source, target = action[0], action[1]
    if source not in planets or target not in planets or source == target:
        return NOOP

    source = planets[source]
    target = planets[target]

    ships = source["ships"]
    ship_ratios = action[2]

    num_a = int(round(ship_ratios[0] * ships[0]))
    num_b = int(round(ship_ratios[1] * ships[1]))
    num_c = int(round(ship_ratios[2] * ships[2]))

    if num_a == 0 and num_b == 0 and num_c == 0:
        return NOOP
    if num_a < 0 or num_b < 0 or num_c < 0:
        print("ship ratios and ships:", ship_ratios, ships)
    return f'send {source["id"]} {target["id"]} {num_a} {num_b} {num_c}'

def parse_game_state(game_state):
    return feed_dict(game_state)
