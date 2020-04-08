# pylint: disable=missing-docstring

import unittest

import numpy as np

from state import feed_dict
from state import FLEET_FEATURE_COUNT, FLEETS
from state import PLANETS, PLANETS_COUNT
from state import HYPERLANE_SOURCES, HYPERLANE_TARGETS
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

        # every hyperlane counts as an empty fleet
        self.assertEqual(
            mapped[FLEETS].shape,
            (len(hyperlanes), FLEET_FEATURE_COUNT))

    # pylint: disable=too-many-locals
    def test_feed_dict(self):
        planets = self.planets

        fleets = self.fleets
        fleets_encoded_permutation_1 = np.asarray([
            # Hyperlane (1, 0)
            [0, 0, 0, 0, 0],
            # Hyperlane (1, 0), fleet #0
            [
                fleets[0].owner - 1,
                fleets[0].eta,
                fleets[0].ships[0],
                fleets[0].ships[1],
                fleets[0].ships[2],
            ],
            # Hyperlane (0, 1)
            [0, 0, 0, 0, 0],
        ], dtype=np.int32)
        sources_encoded_permutation_1 = np.asarray([
            1,  # Hyperlane 1
            1,  # Hyperlane 1, fleet #0
            0,  # Hyperlane 2
        ])
        targets_encoded_permutation_1 = np.asarray([
            0,  # Hyperlane 1
            0,  # Hyperlane 1, fleet #0
            1,  # Hyperlane 2
        ])

        fleets_encoded_permutation_2 = np.asarray([
            # Hyperlane 2
            [0, 0, 0, 0, 0],
            # Hyperlane 1
            [0, 0, 0, 0, 0],
            [
                fleets[0].owner,
                fleets[0].eta,
                fleets[0].ships[0],
                fleets[0].ships[1],
                fleets[0].ships[2],
            ]
        ], dtype=np.int32)
        sources_encoded_permutation_2 = np.asarray([
            0,  # Hyperlane 2
            1,  # Hyperlane 1
            1,  # Hyperlane 1, fleet #0
        ])
        targets_encoded_permutation_2 = np.asarray([
            1,  # Hyperlane 1
            1,  # Hyperlane 1, fleet #0
            0,  # Hyperlane 2
        ])

        hyperlanes = self.hyperlanes

        state = State(
            hyperlanes=hyperlanes,
            planets=planets)

        record = feed_dict(state)
        np.testing.assert_equal(
            record[PLANETS_COUNT],
            np.asarray([len(planets)]))
        np.testing.assert_almost_equal(
            record[PLANETS],
            np.asarray([
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
            ], dtype=np.float32))

        # Hyperlanes order is not defined
        edges = record[FLEETS]
        src = record[HYPERLANE_SOURCES]
        tgt = record[HYPERLANE_TARGETS]

        match = False
        for exp_src, exp_tgt, exp_edges in [
                (
                    sources_encoded_permutation_1,
                    targets_encoded_permutation_1,
                    fleets_encoded_permutation_1,
                ),
                (
                    sources_encoded_permutation_2,
                    targets_encoded_permutation_2,
                    fleets_encoded_permutation_2,
                )]:
            match = np.array_equal(exp_src, src) and \
                    np.array_equal(exp_tgt, tgt) and \
                    np.array_equal(exp_edges, edges)
            if match:
                break

        if not match:
            self.fail(
                "unexpected edges, edge sources or " +
                "edge targets:\n{}\n{}\n{}".format(edges, src, tgt))
