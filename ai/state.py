"""Utilities for encoding/decoding game state"""
from typing import Mapping, List, Dict

import graph_nets as gn
import tensorflow as tf
from tensorflow.keras import Input  # pylint: disable=import-error

import numpy as np

from models import State, Planet


PLANET_FEATURE_COUNT = 10

FLEET_FEATURE_COUNT = 5  # Without origin and target
HYPERLANE_EMBEDDING_SIZE = 32

NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = "senders"
GLOBALS = "globals"

FLEETS = "fleets"
FLEETS_COUNT = "fleets_count"
PLANETS = "planets"
PLANETS_COUNT = "planets_count"
HYPERLANE_TARGETS = "hyperlane_targets"
HYPERLANE_SOURCES = "hyperlane_sources"
HYPERLANE_FLEET_COUNT = "hyperlane_fleet_count"  # fleets count per hyperlane
HYPERLANE_COUNT = "hyperlanes_count"


def feed_dict(
        states: List[State]) -> Dict[str, np.array]:
    """
    Creates mapping from models to batch of game states.
    """

    mapping: Dict[str, np.array] = {}

    for idx, state in enumerate(states):
        mapping['{}_{}'.format(PLANETS_COUNT, idx)] = np.asarray([
            len(state.planets)], dtype=np.int32)
        mapping['{}_{}'.format(HYPERLANE_COUNT, idx)] = np.asarray([
            len(state.hyperlanes)], dtype=np.int32)

        mapping['{}_{}'.format(PLANETS, idx)] = _encode_planets(state.planets)

        fleets = []
        sources = np.zeros(
            shape=(len(state.hyperlanes)),
            dtype=np.int32)
        targets = np.zeros_like(sources)
        hyperlane_fleet_count = np.zeros_like(sources, dtype=np.int32)
        hyperlane_idx = 0
        for from_to, hyperlane in state.hyperlanes.items():
            sources[hyperlane_idx] = from_to[0]
            targets[hyperlane_idx] = from_to[1]
            hyperlane_fleet_count[hyperlane_idx] = len(hyperlane.fleets)
            for fleet in hyperlane.fleets:
                fleets.append([
                    fleet.owner,
                    fleet.eta,
                    fleet.ships[0],
                    fleet.ships[1],
                    fleet.ships[2],
                ])
            hyperlane_idx += 1

        mapping['{}_{}'.format(HYPERLANE_SOURCES, idx)] = sources
        mapping['{}_{}'.format(HYPERLANE_TARGETS, idx)] = targets
        mapping['{}_{}'.format(
            HYPERLANE_FLEET_COUNT, idx)] = hyperlane_fleet_count

        if fleets:
            fleets = np.asarray(fleets, dtype=np.float32)
        else:
            fleets = np.zeros(shape=(0, FLEET_FEATURE_COUNT), dtype=np.float32)
        mapping['{}_{}'.format(FLEETS, idx)] = fleets
        mapping['{}_{}'.format(FLEETS_COUNT, idx)] = np.asarray([
            len(fleets)])

    return mapping


def state_inputs2graphs_tuple(
        state_inputs: List[Mapping[str, Input]]) -> gn.graphs.GraphsTuple:
    """
    Transforms model inputs from create_state_inputs() into
    GraphsTuple objects, which can be used for encoding the game state.
    """
    data_dicts = []

    to_ragged = tf.keras.layers.Lambda(
        lambda args: tf.RaggedTensor.from_row_lengths(
            values=args[0], row_lengths=args[1]).to_tensor(),
        name='to_ragged')
    mask_zeros = tf.keras.layers.Masking(name='mask_zeros')
    lane_embedder = tf.keras.layers.TimeDistributed(
        layer=tf.keras.layers.Dense(
            units=HYPERLANE_EMBEDDING_SIZE,
            name='fleet_embedder'),
        name='lane_embedder')

    for state in state_inputs:
        # Edges
        lanes_count = state[HYPERLANE_COUNT]
        lanes_count = tf.reshape(lanes_count, ())

        # Create edge embeddings from fleets
        lanes = to_ragged([
            state[FLEETS],
            state[HYPERLANE_FLEET_COUNT],
        ])
        lanes = mask_zeros(lanes)
        lanes = lane_embedder(lanes)
        # We have explicitly specify tensor shape, because
        # TimeDistributed will output wrong dimensions when
        # there are 0 elements in a sequence.
        lanes = tf.reshape(
            lanes,
            shape=(lanes_count, -1, HYPERLANE_EMBEDDING_SIZE))
        lanes = _safe_mean(lanes, axis=1)
        lanes = tf.reshape(
            lanes,
            shape=(lanes_count, HYPERLANE_EMBEDDING_SIZE),
            name='reshape_lanes')

        # Nodes
        planets = state[PLANETS]
        planets_count = state[PLANETS_COUNT]
        planets_count = tf.reshape(planets_count, ())
        planets_feature_count = planets.shape[1]
        planets = tf.reshape(
            planets,
            shape=(planets_count, planets_feature_count))

        data_dicts.append({
            GLOBALS: tf.zeros(shape=(1,)),
            NODES: planets,
            EDGES: lanes,
            SENDERS: tf.reshape(
                state[HYPERLANE_SOURCES],
                shape=(lanes_count,)),
            RECEIVERS: tf.reshape(
                state[HYPERLANE_TARGETS],
                shape=(lanes_count,)),
        })

    return gn.utils_tf.data_dicts_to_graphs_tuple(data_dicts)


def _encode_planets(planets: Mapping[int, Planet]) -> np.array:
    encoded = np.zeros(
        shape=(len(planets), PLANET_FEATURE_COUNT),
        dtype=np.float32)
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
        PLANETS_COUNT: Input(
            shape=(),
            name='{}_{}'.format(PLANETS_COUNT, suffix),
            dtype=tf.dtypes.int32),
        HYPERLANE_COUNT: Input(
            shape=(),
            name='{}_{}'.format(HYPERLANE_COUNT, suffix),
            dtype=tf.dtypes.int32),
        HYPERLANE_SOURCES: Input(
            shape=(),
            name='{}_{}'.format(HYPERLANE_SOURCES, suffix),
            dtype=tf.dtypes.int32),
        HYPERLANE_TARGETS: Input(
            shape=(),
            name='{}_{}'.format(HYPERLANE_TARGETS, suffix),
            dtype=tf.dtypes.int32),
        FLEETS_COUNT: Input(
            shape=(),
            name='{}_{}'.format(FLEETS_COUNT, suffix),
            dtype=tf.dtypes.int32),
        PLANETS: Input(
            shape=(PLANET_FEATURE_COUNT,),
            name='{}_{}'.format(PLANETS, suffix),
            dtype=tf.dtypes.float32),
        FLEETS: Input(
            shape=(FLEET_FEATURE_COUNT,),
            name='{}_{}'.format(FLEETS, suffix),
            dtype=tf.dtypes.float32),
        HYPERLANE_FLEET_COUNT: Input(
            shape=(),
            name='{}_{}'.format(HYPERLANE_FLEET_COUNT, suffix),
            dtype=tf.dtypes.int32),
    }


def _safe_mean(tensor, axis: int):
    """
    Standard keras and tf implementation return nans
    when there are 0 elements.
    """
    return tf.keras.layers.Lambda(
        lambda arg: tf.math.divide_no_nan(
            tf.cast(
                tf.math.reduce_sum(arg, axis=axis),
                dtype=tf.dtypes.float32),
            tf.cast(
                tf.shape(arg)[axis],
                dtype=tf.dtypes.float32)))(tensor)