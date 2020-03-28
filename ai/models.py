"""Models representing game state"""
from typing import NamedTuple, Set, Tuple, Mapping


class Fleet(NamedTuple):
    """Fleet of ships"""
    ships: Tuple[int, int, int]
    origin: int
    target: int
    eta: int
    owner: int


class Planet(NamedTuple):
    """Planet"""
    x: int
    y: int
    owner: int
    ships: Tuple[int, int, int]
    production: Tuple[int, int, int]
    production_rounds_left: int


class Hyperlane(NamedTuple):
    """Hyperlane connecting two planets"""
    origin: int
    target: int
    fleets: Tuple[Fleet, ...] = ()
    action: Tuple[int, int, int] = (0, 0, 0)


class State(NamedTuple):
    """Complete snapshot of the game state"""
    planets: Mapping[int, Planet]
    hyperlanes: Set[Hyperlane]
