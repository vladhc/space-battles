"""Models representing game state"""
from typing import NamedTuple, Tuple, Mapping


class Fleet(NamedTuple):
    """Fleet of ships"""
    ships: Tuple[int, int, int]
    origin: int
    target: int
    eta: int
    owner: int


class Planet(NamedTuple):
    """Planet"""
    x: int = 0
    y: int = 0
    owner: int = 0
    ships: Tuple[int, int, int] = (0, 0, 0)
    production: Tuple[int, int, int] = (0, 0, 0)
    production_rounds_left: int = 0


class Hyperlane(NamedTuple):
    """Hyperlane connecting two planets"""
    origin: int
    target: int
    fleets: Tuple[Fleet, ...] = ()
    action: Tuple[int, int, int] = (0, 0, 0)


class State(NamedTuple):
    """Complete snapshot of the game state"""
    planets: Mapping[int, Planet]
    hyperlanes: Mapping[Tuple[int, int], Hyperlane]
