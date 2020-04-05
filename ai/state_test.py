# pylint: disable=missing-docstring

import unittest

import numpy as np

from state import feed_dict
from state import FLEET_FEATURE_COUNT, FLEETS, FLEETS_COUNT
from state import PLANETS, PLANETS_COUNT
from state import HYPERLANE_SOURCES, HYPERLANE_TARGETS, HYPERLANE_COUNT
from state import HYPERLANE_FLEET_COUNT
from models import State, Hyperlane, Planet, Fleet


class TestState(unittest.TestCase):

    def setUp(self):
        self.planets = {
            1: Planet(
                owner=1,
                x=2,
                y=3,
                ships=(3, 2, 1),
                production=(6, 5, 1),
                production_rounds_left=2),
            0: Planet(
                owner=0,
                x=4,
                y=6,
                ships=(4, 5, 6))}
        self.fleets = [
            Fleet(  # The only and the one
                owner=2,
                origin=1,
                target=0,
                ships=(3, 2, 1),
                eta=5),
        ]
        self.hyperlanes = {
            (1, 0): Hyperlane(
                origin=1,
                target=0,
                fleets=(self.fleets[0],)),
            (0, 1): Hyperlane(
                origin=0,
                target=1),
        }

    def test_states_with_empty_fleets(self):
        hyperlanes = {
            (0, 1): Hyperlane(
                origin=1,
                target=0),
            (1, 0): Hyperlane(
                origin=0,
                target=1),
        }
        state = State(
            hyperlanes=hyperlanes,
            planets=self.planets)

        mapped = feed_dict(state)

        self.assertEqual(mapped[FLEETS].shape, (0, FLEET_FEATURE_COUNT))

    def test_feed_dict(self):
        planets = self.planets
        planets_encoded = np.asarray([
            [
                planets[0].owner,
                planets[0].x,
                planets[0].y,
                planets[0].ships[0],
                planets[0].ships[1],
                planets[0].ships[2],
                planets[0].production[0],
                planets[0].production[1],
                planets[0].production[2],
                planets[0].production_rounds_left,
            ], [
                planets[1].owner,
                planets[1].x,
                planets[1].y,
                planets[1].ships[0],
                planets[1].ships[1],
                planets[1].ships[2],
                planets[1].production[0],
                planets[1].production[1],
                planets[1].production[2],
                planets[1].production_rounds_left,
            ],
        ], dtype=np.float32)

        fleets = self.fleets
        fleets_encoded = np.asarray([
            [
                fleets[0].owner,
                fleets[0].eta,
                fleets[0].ships[0],
                fleets[0].ships[1],
                fleets[0].ships[2],
            ]
        ], dtype=np.int32)

        hyperlanes = self.hyperlanes

        state = State(
            hyperlanes=hyperlanes,
            planets=planets)

        record = feed_dict(state)
        np.testing.assert_equal(
            record[PLANETS_COUNT],
            np.asarray([len(planets)]))
        np.testing.assert_equal(
            record[HYPERLANE_COUNT],
            np.asarray([len(state.hyperlanes)]))
        np.testing.assert_equal(
            record[FLEETS_COUNT],
            np.asarray([len(fleets)]))
        np.testing.assert_almost_equal(
            record[PLANETS],
            planets_encoded)
        np.testing.assert_almost_equal(
            record[FLEETS],
            fleets_encoded)

        # Hyperlanes order is not defined
        src = record[HYPERLANE_SOURCES]
        tgt = record[HYPERLANE_TARGETS]
        fleet_count = record[HYPERLANE_FLEET_COUNT]
        lanes_match = np.array_equal(
            src, [0, 1]) and np.array_equal(
                tgt, [1, 0] and np.array_equal(
                    fleet_count, [0, 1]))
        lanes_match = lanes_match or (np.array_equal(
            src, [1, 0]) and np.array_equal(
                tgt, [0, 1]) and np.array_equal(
                    fleet_count, [1, 0]))
        if not lanes_match:
            self.fail(
                "unexpected hyperlane_sources, " +
                "hyperlane_targets and " +
                "hyperlane_fleet_count:\n{} {} {}".format(
                    src, tgt, fleet_count))
