"""Models representing game state"""
from typing import NamedTuple, List, Tuple


class Fleet(NamedTuple):
    """Fleet of ships"""
    ships: Tuple[int, int, int]
    eta: int
    owner: int


class Planet(NamedTuple):
    """Planet"""
    id: int
    x: int
    y: int
    owner: int


class Hyperlane(NamedTuple):
    """Hyperlane connecting two planets"""
    sender: int
    receiver: int
    action: Tuple[int, int, int] = (0, 0, 0)
    fleets: List[Fleet] = []


class State(NamedTuple):
    """Complete snapshot of the game state"""
    planets: List[Planet] = []
    hyperlanes: List[Hyperlane] = []
