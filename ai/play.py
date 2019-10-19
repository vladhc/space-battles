import logging
import pickle
import argparse
from collections import deque

import glob
import os
import os.path as path

import tensorflow as tf
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

from adapter import RPSAdapter
from game_client import RPSClient
from env import SpaceEnv


logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO)
logger = logging.getLogger(__name__)


FRAME_STACK = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoint")
    parser.add_argument("--checkpointdir", dest="checkpointdir")
    parser.add_argument("--username", dest="username")
    parser.add_argument("--password", dest="password")
    parser.add_argument("--addr", dest="addr")
    parser.add_argument('--num-games', dest='num_games', default=10)
    parsed, _ = parser.parse_known_args()

    if parsed.checkpoint and parsed.checkpointdir:
        raise(Exception("Can't set both checkpoint and checkpoint dir"))

    return parsed


def main(args):

    checkpoint = None

    if args.checkpointdir:
        path_glob = path.join(args.checkpointdir, '*.cpk')
        list_of_files = glob.glob(path_glob)
        latest_file = max(list_of_files, key=os.path.getctime)
        print('Using checkpoint: ', latest_file)
        checkpoint = latest_file
    else:
        checkpoint = args.checkpoint

    weights = pickle.load(open(checkpoint, "rb"))

    graph = tf.get_default_graph()
    with graph.device("/device:CPU:0"):
        policy = PPOTFPolicy(
            obs_space=SpaceEnv.observation_space,
            action_space=SpaceEnv.action_space,
            config={
                "model": {
                    "fcnet_activation": "tanh",
                    "fcnet_hiddens": [1024, 512, 256, 64],
                    "max_seq_len": 20,
                },
            })
    policy.set_weights(weights)

    adapter = RPSAdapter()

    wcl = RPSClient(
        addr=args.addr,
        username=args.username,
        password=args.password)

    for _ in range(args.num_games):
        done = False
        game_state = wcl.reset()

        while not done:
            game_state = wcl.game_state()
            done = game_state["game_over"]
            print(game_state)
            state = adapter.parse_game_state(game_state)
            action, _, _ = policy.compute_single_action(state, [])
            action = (action[0][0], action[1][0], action[2][0])
            print(action)
            cmd = adapter.to_game_action(action)
            print("sending command:", cmd)
            wcl.action(cmd)


if __name__ == "__main__":
    PARSED_ARGS = parse_args()
    while True:
        main(PARSED_ARGS)
