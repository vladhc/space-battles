"""Utilities for encoding/decoding game state"""
from typing import Mapping, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input

from models import State, Planet


PLANET_FEATURE_COUNT = 10

FLEET_FEATURE_COUNT = 5  # Without origin and target


def map_inputs_to_states(states: List[State]) -> Mapping[Input, np.array]:
    """
    Creates mapping from models to batch of game states.
    """

    mapping = {}

    for idx, state in enumerate(states):
        mapping['planets_count_{}'.format(idx)] = np.asarray([
            len(state.planets)], dtype=np.int32)
        mapping['hyperlanes_count_{}'.format(idx)] = np.asarray([
            len(state.hyperlanes)], dtype=np.int32)

        mapping['planets_{}'.format(idx)] = _encode_planets(state.planets)

        fleets = []
        sources = np.zeros(
            shape=(len(state.hyperlanes)),
            dtype=np.int32)
        targets = np.zeros_like(sources)
        hyperlane_idx = 0
        for from_to, hyperlane in state.hyperlanes.items():
            sources[hyperlane_idx] = from_to[0]
            targets[hyperlane_idx] = from_to[1]
            for fleet in hyperlane.fleets:
                fleets.append([
                    fleet.owner,
                    fleet.eta,
                    fleet.ships[0],
                    fleet.ships[1],
                    fleet.ships[2],
                ])
            hyperlane_idx += 1

        mapping['hyperlane_sources_{}'.format(idx)] = sources
        mapping['hyperlane_targets_{}'.format(idx)] = targets

        mapping['fleets_{}'.format(idx)] = np.asarray(fleets)
        mapping['fleets_count_{}'.format(idx)] = np.asarray([
            len(fleets)])

    return mapping


def _encode_planets(planets: Mapping[int, Planet]) -> np.array:
    encoded = np.zeros(
        shape=(len(planets), PLANET_FEATURE_COUNT),
        dtype=np.int32)
    for planet_id, planet in planets.items():
        encoded[planet_id] = [
            planet.owner,
            planet.x,
            planet.y,
            planet.ships[0],
            planet.ships[1],
            planet.ships[2],
            planet.production[0],
            planet.production[1],
            planet.production[2],
            planet.production_rounds_left,
        ]
    return encoded


def create_state_inputs(batch_size: int) -> List[Mapping[str, Input]]:
    """
    Creates Keras Input-s which are used for feeding data into the model
    """
    return [
        _create_state_input(str(idx))
        for idx in range(batch_size)
    ]


def _create_state_input(suffix: str) -> Mapping[str, Input]:
    return {
        'planets_count': Input(
            shape=(),
            name='planets_count_{}'.format(suffix),
            dtype=tf.dtypes.int32),
        'hyperlanes_count': Input(
            shape=(),
            name='hyperlanes_count_{}'.format(suffix),
            dtype=tf.dtypes.int32),
        'hyperlane_sources': Input(
            shape=(),
            name='hyperlanes_sources_{}'.format(suffix),
            dtype=tf.dtypes.int32),
        'hyperlane_targets': Input(
            shape=(),
            name='hyperlanes_targets_{}'.format(suffix),
            dtype=tf.dtypes.int32),
        'fleets_count': Input(
            shape=(),
            name='fleets_count_{}'.format(suffix),
            dtype=tf.dtypes.int32),
        'planets': Input(
            shape=(PLANET_FEATURE_COUNT,),
            name='planets_{}'.format(suffix),
            dtype=tf.dtypes.float32),
        'fleets': Input(
            shape=(FLEET_FEATURE_COUNT,),
            name='fleets_{}'.format(suffix),
            dtype=tf.dtypes.float32),
    }
