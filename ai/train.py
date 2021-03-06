# pylint: disable=missing-docstring
from time import time
import random
from typing import List, Set, Tuple, Dict
import itertools
from collections import defaultdict

import sonnet as snt
import graph_nets as gn
import tensorflow as tf
import numpy as np

from models import State
from model import StateEncoder
from dataset import states_gen, states_to_graphs


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


def create_action(
        state: State,
        correct: bool) -> Tuple[float, float, float, float]:
    lanes = set(state.hyperlanes)
    if correct:
        lane = random.sample(list(lanes), k=1)[0]
    else:
        lane = _create_incorrect_lane(lanes, len(state.planets))
    return (
        float(state.planets[lane[0]].x),
        float(state.planets[lane[0]].y),
        float(state.planets[lane[1]].x),
        float(state.planets[lane[1]].y),
    )


def _fleet_labels(state: State) -> Tuple[int, int, int]:
    ships = [0, 0, 0]
    for hyperlane in state.hyperlanes.values():
        for fleet in hyperlane.fleets:
            if fleet.owner == 1:
                ships[0] += fleet.ships[0]
                ships[1] += fleet.ships[1]
                ships[2] += fleet.ships[2]
    return (ships[0], ships[1], ships[2])


def create_dataset_generator(batch_size):
    for states_batch in states_gen(batch_size):
        actions = np.zeros((batch_size, 4), dtype=np.float32)
        labels = np.zeros((batch_size,), dtype=np.int32)
        fleet_labels = np.zeros((batch_size, 3), dtype=np.int32)
        for idx, state in enumerate(states_batch):
            labels[idx] = random.randint(0, 1)
            actions[idx] = create_action(state, labels[idx])
            fleet_labels[idx] = _fleet_labels(state)

        yield {
            "graphs": states_to_graphs(states_batch)._asdict(),
            "actions": actions,
            "labels": labels,
            "fleet_labels": fleet_labels,
        }


def create_dataset(batch_size: int) -> tf.data.Dataset:
    example_graph = states_to_graphs(next(states_gen(batch_size)))
    graphs_spec = gn.utils_tf.specs_from_graphs_tuple(example_graph)

    output_types = {
        "graphs": {
            prop: spec.dtype for prop, spec in graphs_spec._asdict().items()
        },
        "actions": tf.float32,
        "labels": tf.int32,
        "fleet_labels": tf.int32,
    }
    output_shapes = {
        "graphs": {
            prop: spec.shape for prop, spec in graphs_spec._asdict().items()
        },
        "actions": tf.TensorShape((batch_size, 4)),
        "labels": tf.TensorShape((batch_size,)),
        "fleet_labels": tf.TensorShape((batch_size, 3)),
    }

    dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset_generator(batch_size),
        output_types=output_types,
        output_shapes=output_shapes)

    return dataset


def _compile_train_step(model: snt.Module, dataset: tf.data.Dataset):
    optimizer = snt.optimizers.Adam(learning_rate=0.001)
    loss_fn_actions = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_fn_fleets = tf.keras.losses.MeanSquaredError()

    def _train_step(graphs, actions, labels, fleet_labels):
        with tf.GradientTape() as tape:
            out, logits, fleets = model(graphs, actions)
            loss_actions = loss_fn_actions(y_true=labels, y_pred=logits)
            loss_fleets = loss_fn_fleets(y_true=fleet_labels, y_pred=fleets)
            loss = loss_actions + loss_fleets
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        return out, loss_actions, loss_fleets

    input_signature = [
        dataset.element_spec["graphs"],
        dataset.element_spec["actions"],
        dataset.element_spec["labels"],
        dataset.element_spec["fleet_labels"],
    ]
    return tf.function(
        _train_step,
        input_signature=input_signature)


def train(
        model: snt.Module,
        dataset: tf.data.Dataset,
        steps_per_epoch: int = 100):

    compiled_train_step = _compile_train_step(model, dataset)

    # Configure callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            "logs",
            write_graph=False,
            profile_batch=0),
    ]
    metric_actions = tf.keras.metrics.BinaryAccuracy()
    for callback in callbacks:
        # Callback needs a model. Otherwise it refuses to work.
        callback.set_model(
            tf.keras.Sequential([
                tf.keras.layers.Dense(1, input_shape=(1,)),
            ]))
        callback.on_train_begin()

    dataset_iter = iter(dataset)

    for epoch in itertools.count():
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
        measures = Measures()
        metric_actions.reset_states()

        for _ in range(0, steps_per_epoch):
            measures.start_timer(Measures.STEP_TIME)
            measures.start_timer(Measures.SAMPLE_TIME)
            record = next(dataset_iter)
            measures.stop_timer(Measures.SAMPLE_TIME)

            # Train step
            measures.start_timer(Measures.TRAIN_TIME)
            out, loss_actions, loss_fleets = compiled_train_step(
                record["graphs"],
                record["actions"],
                record["labels"],
                record["fleet_labels"])
            measures.stop_timer(Measures.TRAIN_TIME)

            metric_actions.update_state(record["labels"], out)

            measures.record(Measures.LOSS, loss_actions.numpy())
            measures.record(Measures.LOSS_FLEETS, loss_fleets.numpy())
            measures.stop_timer(Measures.STEP_TIME)

        for callback in callbacks:
            callback.on_epoch_end(
                epoch,
                logs={
                    "loss": measures.avg(Measures.LOSS),
                    "loss_fleets": measures.avg(Measures.LOSS_FLEETS),
                    "accuracy": metric_actions.result().numpy(),
                    "steps_per_sec": measures.avg_speed(Measures.STEP_TIME),
                    "batches_per_sec": measures.avg_speed(
                        Measures.SAMPLE_TIME),
                    "train_steps_per_sec": measures.avg_speed(
                        Measures.TRAIN_TIME),
                })

    for callback in callbacks:
        callback.on_train_end()


class Measures:

    STEP_TIME = "step_time"
    SAMPLE_TIME = "sample_time"
    TRAIN_TIME = "train_time"
    LOSS = "loss"
    LOSS_FLEETS = "loss_fleets"
    ACCURACY = "accuracy"

    def __init__(self):
        self._measures: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, float] = {}

    def record(self, name: str, value: float) -> None:
        self._measures[name].append(value)

    def start_timer(self, name: str) -> None:
        self._timers[name] = time()

    def stop_timer(self, name: str) -> None:
        self.record(name, time() - self._timers[name])
        del self._timers[name]

    def avg_speed(self, name: str) -> float:
        times = self._measures[name]
        total = sum(times)
        if not total:
            return 0
        return len(times) / total

    def avg(self, name: str) -> float:
        values = self._measures[name]
        if not values:
            return 0
        return sum(values) / len(values)


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print("Num GPUs Available: ", len(gpus))
    for gpu in gpus:
        print(gpu)
    assert gpus

    dataset = create_dataset(batch_size=32).repeat().shuffle(100).prefetch(2)
    model = StateEncoder()
    train(model, dataset, 100)


if __name__ == "__main__":
    main()
