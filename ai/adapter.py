import math

import numpy as np


MAX_PLANETS = 20
MAX_FLEETS_PER_PLANET = 10

RELATION_FEATURES_PER_PLANET = 2
# exists, distance


FEATURES_PER_FLEET = 5
# distance, ships(x3), owner

FEATURES_PER_PLANET = 9 + RELATION_FEATURES_PER_PLANET * MAX_PLANETS + \
        MAX_FLEETS_PER_PLANET * FEATURES_PER_FLEET
# x, y, owner, ships(x3), production(x3)


class FeatureArray():

    def __init__(self, arr: np.array):
        self.arr = arr
        self._row = 0
        self._col = 0

    def push(self, value):
        self.arr[self._row, self._col] = value

    def finish_row(self):
        self._row += 1
        self._col = 0


class RPSAdapter():

    def to_game_action(self, action):
        source = action[0]
        target = action[1]
        num_a = action[2]
        num_b = action[3]
        num_c = action[4]
        should_pass = action[5]
        if should_pass > 0.5:
            return 'nop'
        return 'send {} {} {} {} {}'.format(source, target, num_a, num_b, num_c)

    def parse_game_state(self, game_state):
        my_id = RPSAdapter._my_id(game_state)

        state = FeatureArray(
            np.zeros((MAX_PLANETS, FEATURES_PER_PLANET), dtype=np.int))

        planets = {planet["id"]: planet for planet in game_state["planets"]}

        for planet_id, planet in planets.items():
            state.push(planet['x'])
            state.push(planet['y'])
            state.push(1 if planet["owner_id"] == my_id else -1)
            # 3 4 5 -> ships
            for idx, ship in enumerate(planet["ships"]):
                state.push(ship)
            # 6 7 8 -> production
            for idx, prod in enumerate(planet["production"]):
                state.push(prod)

            # 9 ... -> relation to other planets
            for idx in range(MAX_PLANETS):
                skip = idx not in planets or idx == planet_id
                if skip:
                    state.push(0)  # exists flag
                    state.push(0)  # distance
                    continue
                target_planet = planets[idx]
                # "exists" flag
                state.push(1)
                # distance
                state.push(RPSAdapter.distance(planet, target_planet))

            RPSAdapter._fill_fleets(planet_id, game_state, state)
            state.finish_row()

        return state.arr.flatten()

    @staticmethod
    def _fill_fleets(planet_id, game_state, state):
        my_id = RPSAdapter._my_id(game_state)
        fleets = [fleet for fleet in game_state["fleets"] if fleet["target"] == planet_id]
        fleets = sorted(fleets, key=lambda fleet: fleet["eta"])

        for idx, fleet in enumerate(fleets):
            if idx > MAX_FLEETS_PER_PLANET:
                return
            state.push(1 if fleet["owner_id"] == my_id else -1)
            for ship in fleet["ship"]:
                state.push(ship)
            state.push(fleet["eta"] - game_state["round"])

    @staticmethod
    def _my_id(state):
        for player in state["players"]:
            if player["itsme"]:
                return player["id"]
        return None

    @staticmethod
    def distance(planet, other_planet):  # copy-paste from engine
        xdiff = planet["x"] - other_planet["x"]
        ydiff = planet["y"] - other_planet["y"]
        return int(math.ceil(math.sqrt(xdiff*xdiff + ydiff*ydiff)))
