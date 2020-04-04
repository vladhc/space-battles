# pylint: disable=missing-docstring
import json
import random
from typing import List, Deque, Any, Generator, Mapping, Set, Tuple
from collections import deque

import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np

from models import State, Planet, Hyperlane
from state import create_state_inputs
from state import feed_dict
from state import state_inputs2graphs_tuple
from state import PLANETS, PLANETS_COUNT
from adapters import json2state


def _parse_game(game: List[Mapping[str, Any]]) -> List[State]:
    return [json2state(tick['state']) for tick in game]


def load_games() -> Generator[List[State], None, None]:
    while True:
        with open("games.njson", "r") as games_njson:
            for game_json in games_njson:
                yield _parse_game(json.loads(game_json))


def create_model(batch_size: int) -> tf.keras.Model:
    state_inputs = create_state_inputs(
        batch_size=batch_size)
    action_input = tf.keras.Input(
        shape=(2,),
        batch_size=batch_size,
        name='action',
        dtype=tf.dtypes.int32)
    states = []
    for idx, state_input in enumerate(state_inputs):
        planets_count = state_input[PLANETS_COUNT]
        planets_count = tf.reshape(planets_count, ())

        marker_planet_from = tf.one_hot(
            action_input[idx, 0], planets_count,
            dtype=state_input[PLANETS].dtype,
            name="marker_planet_from")
        marker_planet_from = tf.reshape(
            marker_planet_from, (planets_count, 1),
            name="marker_planet_from_reshape")
        marker_planet_to = tf.one_hot(
            action_input[idx, 1], planets_count,
            dtype=state_input[PLANETS].dtype,
            name="marker_planet_to")
        marker_planet_to = tf.reshape(
            marker_planet_to, (planets_count, 1),
            name="marker_planet_to_reshape")

        state = state_input.copy()
        states.append(state)
        state[PLANETS] = tf.concat(
            [marker_planet_from, marker_planet_to, state_input[PLANETS]],
            axis=1)

    graphs_tuple_tf = state_inputs2graphs_tuple(states)

    graph_network = gn.modules.GraphNetwork(
        edge_model_fn=lambda:
            snt.nets.MLP([64, 32], activate_final=True),
        node_model_fn=lambda:
            snt.nets.MLP([64, 32], activate_final=True),
        global_model_fn=lambda: snt.nets.MLP(
            [128, 64], activate_final=False))
    out_graph = graph_network(graphs_tuple_tf)
    out = out_graph.globals

    out = tf.keras.layers.Dense(
        32, name="corr_action_fc_1", activation="tanh")(out)

    is_correct_action_head = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        name='corr_action_out')(out)

    model_inputs = {
        'actions': action_input,
    }
    for st_inputs in state_inputs:
        for state_input in st_inputs.values():
            model_inputs[state_input.name] = state_input

    # return is_correct_action_head
    model = tf.keras.Model(
        inputs=model_inputs,
        outputs=is_correct_action_head)
    model_debug = tf.keras.Model(
        inputs=model_inputs,
        outputs=states)
    return (model, model_debug)


def _create_incorrect_lane(
        correct_lanes: Set[Tuple[int, int]],
        n_planets: int) -> Tuple[int, int]:
    while True:
        lane = (random.randint(0, n_planets-1), random.randint(0, n_planets-1))
        if lane in correct_lanes:
            continue
        if lane[0] == lane[1]:
            continue
        return lane


def create_lane_action(
        state: State,
        correct: bool) -> Tuple[int, int]:
    lanes = set(state.hyperlanes)
    if correct:
        return random.sample(list(lanes), k=1)[0]
    else:
        return _create_incorrect_lane(lanes, len(state.planets))


def create_action(
        state: State,
        correct: bool) -> Tuple[float, float, float, float]:
    lane = _create_lane(state, correct)
    return (
        float(state.planets[lane[0]].x),
        float(state.planets[lane[0]].y),
        float(state.planets[lane[1]].x),
        float(state.planets[lane[1]].y),
    )


def states_dataset(batch_size: int) -> Generator[List[State], None, None]:
    games = load_games()
    states_buffer: List[State] = []
    buffer_size = batch_size * 5000
    while True:
        fetch_buffer = len(states_buffer) < buffer_size
        if fetch_buffer:
            print("Fetching buffer...")
        while len(states_buffer) < buffer_size:
            states = next(games)  # pylint: disable=stop-iteration-return
            states_buffer.extend(states)
        if fetch_buffer:
            random.shuffle(states_buffer)
        yield states_buffer[:batch_size]
        states_buffer = states_buffer[batch_size:]


def simple_states_dataset(batch_size: int):
    state = State(
        planets={
            0: Planet(x=-10, y=1),
            1: Planet(x=10, y=2),
            2: Planet(x=0, y=10),
        },
        hyperlanes={
            (0, 1): Hyperlane(origin=0, target=1),
            (1, 0): Hyperlane(origin=1, target=0),
        },
    )
    states_batch = [state] * batch_size
    while True:
        yield states_batch


def create_dataset(batch_size):
    states = simple_states_dataset(batch_size)
    while True:
        states_batch = next(states)  # pylint: disable=stop-iteration-return
        actions = np.zeros((batch_size, 2), dtype=np.int32)
        labels = np.zeros((batch_size,), dtype=np.int32)
        for idx, state in enumerate(states_batch):
            labels[idx] = random.randint(0, 1)
            actions[idx] = create_lane_action(state, labels[idx])

        features = feed_dict(states_batch)
        features['action'] = actions
        yield features, labels


def train(models: Tuple[tf.keras.Model], dataset, steps: int):
    # Prepare dataset
    last_acc: Deque[Any] = deque([], 3200)
    last_loss: Deque[Any] = deque([], 3200)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    metric = tf.keras.metrics.BinaryAccuracy()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    model = models[0]
    model_debug = models[1]
    debug = False

    for step in range(0, steps):
        features, labels = next(dataset)
        if debug:
            print("--- step {} ---".format(step))
            print("features:", features["planets_0"], features["action"])
            print("labels:", labels)
            out_debug = model_debug(features)[0][PLANETS]
            print("out_debug:", out_debug)

        # Train step
        with tf.GradientTape() as tape:
            out = model(features)
            loss = loss_fn(labels, out)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metric.reset_states()
        metric.update_state(labels, out)

        last_loss.append(loss.numpy())
        last_acc.append(metric.result().numpy())

        print('step {}: {:.2f} {:.2f} ({:.2f}, {:.2f})'.format(
            step,
            loss.numpy(), metric.result().numpy(),
            sum(last_loss)/len(last_loss),
            sum(last_acc)/len(last_acc)))


def main():
    batch_size = 16
    dataset = create_dataset(batch_size)
    models = create_model(batch_size)
    train(models, dataset, 1000000)


if __name__ == '__main__':
    main()
