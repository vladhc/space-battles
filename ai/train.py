import pickle
import argparse

import ray
import ray.tune as tune

from env import create_env
from ray.tune.registry import register_env


def run():
    register_env("SpaceEnv", create_env)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore",
        dest="restore",
        help="Path to the previous training directory")
    args, _ = parser.parse_known_args()

    ray.init()
    train(args.restore)


def train(restore):

    def _on_train_result(info):
        result = info["result"]
        iteration = result["training_iteration"]

        trainer = info["trainer"]
        policy = trainer.get_policy()
        weights = policy.get_weights()

        pickle.dump(weights, open("checkpoint-{}.ckpt".format(iteration), "wb"))

    tune.run(
        "PPO",
        stop={"timesteps_total": 40000 * 8000000000},
        checkpoint_freq=10,
        restore=restore,
        config={
            "env": "SpaceEnv",
            "num_gpus": 1,
            "num_workers": 18,
            "num_envs_per_worker": 2,
            "train_batch_size": 128,
            "sample_batch_size": 128,
            "num_envs_per_worker": 1,
            "lr": 2e-4,
            "model": {
                "fcnet_activation": "tanh",
                "fcnet_hiddens": [512, 256, 64],
            },
            "callbacks": {
                "on_train_result": _on_train_result,
            },
        },
    )


if __name__ == "__main__":
    run()
