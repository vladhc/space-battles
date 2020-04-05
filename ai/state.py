"""Utilities for encoding/decoding game state"""
from typing import Mapping, Dict

import numpy as np

from models import State, Planet


PLANET_FEATURE_COUNT = 10

FLEET_FEATURE_COUNT = 5  # Without origin and target

FLEETS = "fleets"
FLEETS_COUNT = "fleets_count"
PLANETS = "planets"
PLANETS_COUNT = "planets_count"
HYPERLANE_TARGETS = "hyperlane_targets"
HYPERLANE_SOURCES = "hyperlane_sources"
HYPERLANE_FLEET_COUNT = "hyperlane_fleet_count"  # fleets count per hyperlane
HYPERLANE_COUNT = "hyperlanes_count"


def feed_dict(
        state: State) -> Dict[str, np.array]:
    """
    Creates mapping from models to batch of game states.
    """

    mapping: Dict[str, np.array] = {}

    mapping[PLANETS_COUNT] = np.asarray([
        len(state.planets)], dtype=np.int32)
    mapping[HYPERLANE_COUNT] = np.asarray([
        len(state.hyperlanes)], dtype=np.int32)

    mapping[PLANETS] = _encode_planets(state.planets)

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

    mapping[HYPERLANE_SOURCES] = sources
    mapping[HYPERLANE_TARGETS] = targets
    mapping[HYPERLANE_FLEET_COUNT] = hyperlane_fleet_count

    if fleets:
        fleets = np.asarray(fleets, dtype=np.float32)
    else:
        fleets = np.zeros(shape=(0, FLEET_FEATURE_COUNT), dtype=np.float32)
    mapping[FLEETS] = fleets
    mapping[FLEETS_COUNT] = np.asarray([len(fleets)], dtype=np.int32)

    return mapping


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
