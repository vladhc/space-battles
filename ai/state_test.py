# pylint: disable=missing-docstring

import unittest

import numpy as np
import tensorflow as tf

from state import create_state_inputs, map_inputs_to_states
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
            (0, 1): Hyperlane(
                origin=1,
                target=0,
                fleets=(self.fleets[0],)),
            (1, 0): Hyperlane(
                origin=0,
                target=1),
        }

    def test_state_inputs(self):
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

        inputs = create_state_inputs(batch_size=1)
        model = tf.keras.Model(inputs=inputs, outputs=inputs)

        output = model.predict_on_batch(
            map_inputs_to_states([state]))

        self.assertEqual(len(output), 1)
        record = output[0]
        np.testing.assert_equal(
            record['planets_count'],
            np.asarray([len(planets)]))
        np.testing.assert_equal(
            record['hyperlanes_count'],
            np.asarray([len(state.hyperlanes)]))
        np.testing.assert_equal(
            record['fleets_count'],
            np.asarray([len(fleets)]))
        np.testing.assert_almost_equal(
            record['planets'],
            planets_encoded)
        np.testing.assert_almost_equal(
            record['fleets'],
            fleets_encoded)

        # Hyperlanes order is not defined
        src = record['hyperlane_sources']
        tgt = record['hyperlane_targets']
        lanes_match = np.array_equal(
            src, [0, 1]) and np.array_equal(tgt, [1, 0])
        lanes_match = lanes_match or (np.array_equal(
            src, [1, 0]) and np.array_equal(tgt, [0, 1]))
        if not lanes_match:
            self.fail(
                "unexpected hyperlane_sources " +
                "and hyperlane_targets:\n{} {}".format(
                    src, tgt))
