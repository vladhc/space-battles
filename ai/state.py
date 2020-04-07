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


def feed_dict(
        state: State) -> Dict[str, np.array]:
    """
    Creates mapping from models to batch of game states.
    """

    mapping: Dict[str, np.array] = {}

    mapping[PLANETS_COUNT] = np.asarray([
        len(state.planets)], dtype=np.int32)

    mapping[PLANETS] = _encode_planets(state.planets)

    edges: List[np.array] = []
    sources: List[int] = []
    targets: List[int] = []
    for from_to, hyperlane in state.hyperlanes.items():
        # Edge which means "hyperlane"
        sources.append(from_to[0])
        targets.append(from_to[1])
        edges.append(np.zeros(FLEET_FEATURE_COUNT, dtype=np.float32))
        # Adding one edge per each fleet
        for fleet in hyperlane.fleets:
            sources.append(from_to[0])
            targets.append(from_to[1])
            edges.append([
                fleet.owner - 1,  # fleet.owner is either 1 or 2
                fleet.eta,
                fleet.ships[0],
                fleet.ships[1],
                fleet.ships[2],
            ])

    mapping[HYPERLANE_SOURCES] = np.asarray(sources, dtype=np.int32)
    mapping[HYPERLANE_TARGETS] = np.asarray(targets, dtype=np.int32)
    mapping[FLEETS] = np.asarray(edges, dtype=np.float32)

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
