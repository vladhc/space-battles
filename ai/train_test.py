# pylint: disable=missing-docstring
import unittest

from train import create_action
from models import State, Hyperlane, Planet


class TestTrain(unittest.TestCase):

    def test_create_action(self):
        state = State(
            planets={
                0: Planet(x=-10, y=1),
                1: Planet(x=10, y=2),
                2: Planet(x=0, y=10),
            },
            hyperlanes={
                (0, 1): Hyperlane(origin=0, target=1),
                (1, 0): Hyperlane(origin=1, target=0),
            },
        )
        valid_actions = {
            (-10, 1, 10, 2),
            (10, 2, -10, 1),
        }
        invalid_actions = {
            (-10, 1, 0, 10),
            (0, 10, -10, 1),
            (0, 10, 10, 2),
            (10, 2, 0, 10),
        }
        self.assertFalse(valid_actions.intersection(invalid_actions))
        for _ in range(100):
            action = create_action(state, correct=True)
            self.assertIn(action, valid_actions)
        for _ in range(100):
            action = create_action(state, correct=False)
            self.assertIn(action, invalid_actions)
