# pylint: disable=missing-docstring

import unittest

import numpy as np
import tensorflow as tf

from state import create_state_inputs, map_inputs_to_states
from models import State, Hyperlane, Planet, Fleet


class TestStateInputs(unittest.TestCase):

    def test_feed(self):
        planets = {
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

        fleets = [
            Fleet(  # The only and the one
                owner=2,
                origin=1,
                target=0,
                ships=(3, 2, 1),
                eta=5),
        ]
        fleets_encoded = np.asarray([
            [
                fleets[0].owner,
                fleets[0].eta,
                fleets[0].ships[0],
                fleets[0].ships[1],
                fleets[0].ships[2],
            ]
        ], dtype=np.int32)

        hyperlanes = {
            (0, 1): Hyperlane(
                origin=1,
                target=0,
                fleets=(fleets[0],)),
            (1, 0): Hyperlane(
                origin=0,
                target=1),
        }
        hyperlane_sources = np.asarray([0, 1], dtype=np.int32)
        hyperlane_targets = np.asarray([1, 0], dtype=np.int32)

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
        np.testing.assert_equal(
            record['hyperlane_sources'],
            hyperlane_sources)
        np.testing.assert_equal(
            record['hyperlane_targets'],
            hyperlane_targets)
