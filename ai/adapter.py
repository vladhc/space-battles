import math

import numpy as np


MAX_PLANETS = 21
MAX_FLEETS_PER_PLANET = 15

RELATION_FEATURES_PER_PLANET = 2
# exists, distance


FEATURES_PER_FLEET = 5
# distance, ships(x3), owner

FEATURES_PER_PLANET = 9 + RELATION_FEATURES_PER_PLANET * MAX_PLANETS + \
        MAX_FLEETS_PER_PLANET * FEATURES_PER_FLEET
# x, y, owner, ships(x3), production(x3)

NOOP = "nop"


class FeatureArray():

    def __init__(self, arr: np.array):
        self.arr = arr
        self._row = 0
        self._col = 0

    def push(self, value):
        self.arr[self._row, self._col] = value
        self._col += 1

    def finish_row(self):
        self._row += 1
        self._col = 0


class RPSAdapter():

    def __init__(self):
        self._last_game_state = None

    def reset(self):
        self._last_game_state = None

    def to_game_action(self, action):
        planets = {planet["id"]: planet for planet in self._last_game_state["planets"]}
        source, target = action[0], action[1]
        if source not in planets or target not in planets or source == target:
            return NOOP
        source = planets[source]
        target = planets[target]

        ships = source["ships"]
        ships_ratio = action[2]

        num_a = int(round(ships_ratio[0] * ships[0]))
        num_b = int(round(ships_ratio[1] * ships[1]))
        num_c = int(round(ships_ratio[2] * ships[2]))
        if num_a == 0 and num_b == 0 and num_c == 0:
            return NOOP
        if num_a < 0 or num_b < 0 or num_c < 0:
            print("ship ratios and ships:", ships_ratio, ships)
        return 'send {} {} {} {} {}'.format(
            source["id"], target["id"],
            num_a, num_b, num_c)

    def parse_game_state(self, game_state):
        self._last_game_state = game_state
        player_id = my_id(game_state)

        state = FeatureArray(
            np.zeros((MAX_PLANETS, FEATURES_PER_PLANET), dtype=np.float32))

        planets = {planet["id"]: planet for planet in game_state["planets"]}

        for planet_id, planet in planets.items():
            state.push(planet['x'])
            state.push(planet['y'])

            owner = 0
            if planet["owner_id"] == player_id:
                owner = 1
            elif planet["owner_id"] != 0:
                owner = -1
            state.push(owner)

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
                else:
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
        player_id = my_id(game_state)
        fleets = [fleet for fleet in game_state["fleets"] if fleet["target"] == planet_id]
        fleets = sorted(fleets, key=lambda fleet: fleet["eta"])

        for idx, fleet in enumerate(fleets):
            if idx >= MAX_FLEETS_PER_PLANET:
                print("max fleets per planet is reached. Total:", len(fleets))
                return
            state.push(1 if fleet["owner_id"] == player_id else -1)
            for ship in fleet["ships"]:
                state.push(ship)
            state.push(fleet["eta"] - game_state["round"])

    @staticmethod
    def distance(planet, other_planet):  # copy-paste from engine
        xdiff = planet["x"] - other_planet["x"]
        ydiff = planet["y"] - other_planet["y"]
        return int(math.ceil(math.sqrt(xdiff*xdiff + ydiff*ydiff)))

def my_id(state):
    for player in state["players"]:
        if player["itsme"]:
            return player["id"]
    return None
