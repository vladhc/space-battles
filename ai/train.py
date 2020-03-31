# pylint: disable=missing-docstring
import json
import random
from typing import List, Deque, Any, Generator, Mapping, Set, Tuple
from collections import deque

import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np

from models import State
from state import create_state_inputs
from state import feed_dict
from state import state_inputs2graphs_tuple
from adapters import json2state, attach_action


def _parse_game(game: List[Mapping[str, Any]]) -> List[State]:
    states: List[State] = []

    for tick in game:
        state = json2state(tick['state'])
        for action in ['action_1', 'action_2']:
            if action in tick:
                state = attach_action(state, tick[action])
        states.append(state)
    return states


def load_games() -> Generator[List[State], None, None]:
    while True:
        with open("games.njson", "r") as games_njson:
            for game_json in games_njson:
                yield _parse_game(json.loads(game_json))


def create_model(batch_size: int) -> tf.keras.Model:
    state_inputs = create_state_inputs(batch_size=batch_size)
    action_input = tf.keras.Input(
        shape=(4,),
        name='action',
        batch_size=batch_size,
        dtype=tf.dtypes.float32)

    graphs_tuple_tf = state_inputs2graphs_tuple(state_inputs)

    graph_network = gn.modules.GraphNetwork(
        edge_model_fn=lambda: snt.nets.MLP([128, 64, 64], activate_final=True),
        node_model_fn=lambda: snt.nets.MLP([128, 64, 64], activate_final=True),
        global_model_fn=lambda: snt.nets.MLP(
            [128, 64, 64], activate_final=False))
    out_graph = graph_network(graphs_tuple_tf)
    encoded = out_graph.globals

    action = tf.keras.layers.Dense(
        8, name="action_fc_1")(action_input)

    out = tf.keras.layers.concatenate(
        [encoded, action],
        name="concat_state_and_action",
        axis=1)

    out = tf.keras.layers.Dense(
        32, name="corr_action_fc_1", activation="relu")(out)
    out = tf.keras.layers.Dense(
        16, name="corr_action_fc_2", activation='relu')(out)

    is_correct_action_head = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        name='corr_action_out')(out)

    model = tf.keras.Model(inputs={
        'state': state_inputs,
        'action': action_input,
    }, outputs=is_correct_action_head)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


def _create_incorrect_lane(
        correct_lanes: Set[Tuple[int, int]],
        n_planets: int) -> Tuple[int, int]:
    while True:
        lane = (random.randint(0, n_planets-1), random.randint(0, n_planets-1))
        if lane not in correct_lanes:
            return lane


def create_action(state: State, correct: bool) -> Tuple[int, int, int, int]:
    lanes = set(state.hyperlanes)
    if correct:
        lane = random.sample(list(lanes), k=1)[0]
    else:
        lane = _create_incorrect_lane(lanes, len(state.planets))
    return (
        state.planets[lane[0]].x,
        state.planets[lane[0]].y,
        state.planets[lane[1]].x,
        state.planets[lane[1]].y,
    )


def states_dataset(batch_size: int) -> Generator[List[State], None, None]:
    games = load_games()
    states_buffer: List[State] = []
    while True:
        while len(states_buffer) < (batch_size * 1000):
            states = next(games)  # pylint: disable=stop-iteration-return
            states_buffer.extend(states)
            random.shuffle(states_buffer)
        yield states_buffer[:batch_size]
        states_buffer = states_buffer[batch_size:]


def train(steps: int, batch_size: int):
    model = create_model(batch_size)

    # Prepare dataset
    last_acc: Deque[Any] = deque([], 100)
    last_loss: Deque[Any] = deque([], 100)

    states = states_dataset(batch_size)
    for step in range(0, steps):
        states_batch = next(states)

        actions = np.zeros((batch_size, 4))
        labels = np.zeros((batch_size,))
        for idx, state in enumerate(states_batch):
            labels[idx] = random.randint(0, 1)
            actions[idx] = create_action(state, labels[idx])

        features = {
            'state': feed_dict(states_batch),
            'actions': actions,
        }

        metrics = model.train_on_batch(features, labels)
        last_loss.append(metrics[0])
        last_acc.append(metrics[1])

        print('step {}: {} ({}, {})'.format(
            step, metrics,
            sum(last_loss)/len(last_loss),
            sum(last_acc)/len(last_acc)))
        if step % 100 == 0:
            model.save('model.h5')


train(1000000, 32)
