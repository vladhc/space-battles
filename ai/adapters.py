"""Functions for converting from game json to model classes"""
from typing import Mapping, Any

from models import State, Planet, Hyperlane, Fleet


def json2state(state: Mapping[str, Any]) -> State:
    """Converts game state json to state model"""
    planets = {
        props['id']: Planet(
            x=props['x'],
            y=props['y'],
            owner=props['owner_id'],
            ships=props['ships'],
            production=props['production'],
            production_rounds_left=props['production_rounds_left'])
        for props in state['planets']
    }
    fleets = [
        Fleet(
            ships=(
                props['ships'][0],
                props['ships'][1],
                props['ships'][2]),
            eta=props['eta'],
            owner=props['owner'] if 'owner' in props else -1,
            origin=props['origin'],
            target=props['target'])
        for props in state['fleets']
    ]
    hyperlanes = {
        Hyperlane(
            origin=props[0],
            target=props[1],
            fleets=(
                fleet for fleet in fleets
                if fleet.origin == props[0] and fleet.target == props[1]
            ))
        for props in state['hyperlanes']
    }
    return State(planets=planets, hyperlanes=hyperlanes)
