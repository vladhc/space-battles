import pandas as pd
import pprint


tstate = {'planets': [{'id': 0, 'x': 0, 'y': 0, 'owner_id': 0, 'ships': [10, 140, 10], 'production': [0, 13, 0]}, {'id': 1, 'x': -8, 'y': 9, 'owner_id': 1, 'ships': [10, 30, 10], 'production': [0, 2, 0]}, {'id': 2, 'x': 8, 'y': -9, 'owner_id': 2, 'ships': [10, 30, 10], 'production': [0, 2, 0]}, {'id': 3, 'x': 6, 'y': 12, 'owner_id': 0, 'ships': [10, 20, 60], 'production': [0, 1, 5]}, {'id': 4, 'x': -6, 'y': -12, 'owner_id': 0, 'ships': [10, 20, 60], 'production': [0, 1, 5]}, {'id': 5, 'x': 12, 'y': -1, 'owner_id': 0, 'ships': [30, 30, 30], 'production': [2, 2, 2]}, {'id': 6, 'x': -12, 'y': 1, 'owner_id': 0, 'ships': [30, 30, 30], 'production': [2, 2, 2]}, {'id': 7, 'x': 12, 'y': 10, 'owner_id': 0, 'ships': [50, 30, 20], 'production': [4, 2, 1]}, {'id': 8, 'x': -12, 'y': -10, 'owner_id': 0, 'ships': [50, 30, 20], 'production': [4, 2, 1]}, {'id': 9, 'x': -17, 'y': 11, 'owner_id': 0, 'ships': [30, 10, 40], 'production': [2, 0, 3]}, {'id': 10, 'x': 17, 'y': -11, 'owner_id': 0, 'ships': [30, 10, 40], 'production': [2, 0, 3]}, {'id': 11, 'x': 12, 'y': 6, 'owner_id': 0, 'ships': [10, 20, 10], 'production': [0, 1, 0]}, {'id': 12, 'x': -12, 'y': -6, 'owner_id': 0, 'ships': [10, 20, 10], 'production': [0, 1, 0]}, {'id': 13, 'x': 1, 'y': -5, 'owner_id': 0, 'ships': [10, 10, 120], 'production': [0, 0, 11]}, {'id': 14, 'x': -1, 'y': 5, 'owner_id': 0, 'ships': [10, 10, 120], 'production': [0, 0, 11]}, {'id': 15, 'x': -2, 'y': -9, 'owner_id': 0, 'ships': [10, 60, 10], 'production': [0, 5, 0]}, {'id': 16, 'x': 2, 'y': 9, 'owner_id': 0, 'ships': [10, 60, 10], 'production': [0, 5, 0]}, {'id': 17, 'x': -3, 'y': -1, 'owner_id': 0, 'ships': [30, 20, 30], 'production': [2, 1, 2]}, {'id': 18, 'x': 3, 'y': 1, 'owner_id': 0, 'ships': [30, 20, 30], 'production': [2, 1, 2]}, {'id': 19, 'x': 20, 'y': 0, 'owner_id': 0, 'ships': [10, 10, 70], 'production': [0, 0, 6]}, {'id': 20, 'x': -20, 'y': 0, 'owner_id': 0, 'ships': [10, 10, 70], 'production': [0, 0, 6]}], 'fleets': [], 'round': 0, 'max_rounds': 500, 'player_id': 1, 'game_over': False, 'winner': None, 'players': [{'id': 1, 'name': 'test', 'itsme': True}, {'id': 2, 'name': 'random_bot', 'itsme': False}]}

class RPSAdapter():
    def to_game_action(action):
        source = action[0]
        target = action[1]
        num_a = action[2]
        num_b = action[3]
        num_c = action[4]
        should_pass = action[5]
        if should_pass > 0.5:
            return 'nop'
        return 'send {} {} {} {} {}'.format(source, target, num_a, num_b, num_c)

    def parse_game_state(state):
        done = state['game_over']

        reward = 0

        planets = state['planets']
        x = [p['x'] for p in planets]
        y = [p['x'] for p in planets]
        num_a, num_b, num_c = [p['ships'] for p in planets]
        prod_a, prod_b, prod_c = [p['production'] for p in planets]

        df = pd.DataFrame(...)

        game_state =

        return df
