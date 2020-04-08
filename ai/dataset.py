"""Functions and classes for working with data pipelines"""
import json
from typing import List, Any, Generator, Mapping

import numpy as np
import graph_nets as gn
from graph_nets.graphs import NODES, EDGES, RECEIVERS, SENDERS, GLOBALS

from models import State
from adapters import json2state
from state import feed_dict

from state import PLANETS
from state import FLEETS
from state import HYPERLANE_TARGETS
from state import HYPERLANE_SOURCES


def _parse_game(game: List[Mapping[str, Any]]) -> List[State]:
    return [json2state(tick['state']) for tick in game]


def _load_games() -> Generator[List[State], None, None]:
    with open("games.njson", "r") as games_njson:
        for game_json in games_njson:
            yield _parse_game(json.loads(game_json))


def states_gen(batch_size: int) -> Generator[List[State], None, None]:
    """Generator which ouputs random valid game states"""
    games = _load_games()
    states_buffer: List[State] = []
    while True:
        while len(states_buffer) < batch_size:
            try:
                states = next(games)  # pylint: disable=stop-iteration-return
            except StopIteration:
                return
            states_buffer.extend(states)
        yield states_buffer[:batch_size]
        states_buffer = states_buffer[batch_size:]


def states_to_graphs(states: List[State]) -> gn.graphs.GraphsTuple:
    """Transforms a batch of game states into a GraphsTuple"""
    data_dicts: List[Mapping[str, np.array]] = []

    for state in states:
        props = feed_dict(state)
        data_dicts.append({
            GLOBALS: np.zeros((0,), dtype=np.float32),
            NODES: props[PLANETS],
            EDGES: props[FLEETS],
            SENDERS: props[HYPERLANE_SOURCES],
            RECEIVERS: props[HYPERLANE_TARGETS],
        })
    return gn.utils_np.data_dicts_to_graphs_tuple(data_dicts)
