# pylint: disable=missing-docstring

import unittest

from adapters import json2state, attach_action
from models import State, Planet, Fleet, Hyperlane


class Json2StateTest(unittest.TestCase):

    def test_basics(self):
        state_json = {
            "game_over": False,
            "winner": None,
            "round": 2,
            "max_rounds": 500,
            "fleets": [{
                "id": 0,
                "owner_id": 1,
                "origin": 0,
                "target": 1,
                "ships": [3, 2, 1],
                "eta": 45
            }],
            "players": [{
                "id": 1,
                "name": "dividuum",
                "itsme": True
            }, {
                "id": 2,
                "name": "cupe",
                "itsme": False
            }],
            "planets": [{
                "id": 1,
                "owner_id": 1,
                "y": 5,
                "x": 2,
                "ships": [1, 2, 3],
                "production": [1, 5, 6],
                "production_rounds_left": 10,
            }, {
                "id": 0,
                "owner_id": 0,
                "y": 10,
                "x": 20,
                "ships": [10, 20, 30],
                "production": [1, 1, 1],
                "production_rounds_left": 100,
            }],
            "hyperlanes": [
                [0, 1],
                [1, 0],
            ],
        }

        state = json2state(state_json)

        exp_state = State(
            hyperlanes={
                (0, 1): Hyperlane(
                    origin=0,
                    target=1,
                    fleets=(
                        Fleet(
                            owner=1,
                            origin=0,
                            target=1,
                            ships=(3, 2, 1),
                            eta=45),)),
                (1, 0): Hyperlane(
                    origin=1,
                    target=0,
                )},
            planets={
                1: Planet(
                    owner=1,
                    y=5,
                    x=2,
                    ships=(1, 2, 3),
                    production=(1, 5, 6),
                    production_rounds_left=10),
                0: Planet(
                    owner=0,
                    y=10,
                    x=20,
                    ships=(10, 20, 30),
                    production=(1, 1, 1),
                    production_rounds_left=100)})
        self.assertEqual(state.planets, exp_state.planets)
        self.assertEqual(len(state.hyperlanes), len(exp_state.hyperlanes))
        self.assertEqual(len(state.hyperlanes), 2)
        self.assertEqual(state.hyperlanes, exp_state.hyperlanes)
        self.assertEqual(state, exp_state)


class AttachActionTest(unittest.TestCase):

    def test_sent(self):
        init_state = State(
            hyperlanes={
                (0, 1): Hyperlane(origin=0, target=1),
                (1, 0): Hyperlane(origin=1, target=0)},
            planets={
                0: Planet(
                    owner=1,
                    ships=(1, 2, 3),
                    production=(1, 5, 6)),
                1: Planet()})

        state = attach_action(init_state, 'send 0 1 1 3 2\n')

        self.assertEqual(state.planets, init_state.planets)
        self.assertEqual(state.hyperlanes, {
            (0, 1): Hyperlane(
                origin=0,
                target=1,
                action=(1, 3, 2)),
            (1, 0): Hyperlane(origin=1, target=0)})
