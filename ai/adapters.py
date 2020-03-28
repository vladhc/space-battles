"""Functions for converting from game json to model classes"""
from typing import Mapping, Any, Dict, Tuple

from models import State, Planet, Hyperlane, Fleet


def json2state(state: Mapping[str, Any]) -> State:
    """Converts game state json to state model"""
    planets = {
        props['id']: Planet(
            x=props['x'],
            y=props['y'],
            owner=props['owner_id'],
            ships=(
                int(props['ships'][0]),
                int(props['ships'][1]),
                int(props['ships'][2])),
            production=(
                int(props['production'][0]),
                int(props['production'][1]),
                int(props['production'][2])),
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
            owner=props['owner_id'],
            origin=props['origin'],
            target=props['target'])
        for props in state['fleets']
    ]
    hyperlanes = {
        (props[0], props[1]): Hyperlane(
            origin=props[0],
            target=props[1],
            fleets=tuple([
                fleet for fleet in fleets
                if fleet.origin == props[0] and fleet.target == props[1]
            ]))
        for props in state['hyperlanes']
    }
    return State(planets=planets, hyperlanes=hyperlanes)


def attach_action(state: State, action: str) -> State:
    """
    Finds an edge in the state, corresponding to the action
    and attaches action's parameter to it. Action
    is an actual string which is sent to the server. For example:
    sent 1 4 8 1 1
    """
    action = action.strip()
    if not action:
        return state
    params = action.split(' ')
    if params[0] != 'sent':
        return state

    origin = int(params[1])
    target = int(params[2])
    coord = (origin, target)

    hyperlanes: Dict[Tuple[int, int], Hyperlane] = {}
    hyperlanes.update(state.hyperlanes)
    hyperlane = state.hyperlanes[coord]

    del hyperlanes[coord]
    hyperlanes[coord] = hyperlane._replace(
        action=(
            int(params[3]),
            int(params[4]),
            int(params[5])))

    return state._replace(hyperlanes=hyperlanes)
