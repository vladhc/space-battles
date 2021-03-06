"""TensorFlow models"""
import graph_nets as gn
import sonnet as snt
import tensorflow as tf


class StateEncoder(snt.Module):
    """Module which encodes level graph into a vector"""

    def __init__(self):
        super().__init__(name="state_encoder")
        self.graph_network = gn.modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP(
                [32, 32, 16], activate_final=False),
            node_model_fn=lambda: snt.nets.MLP(
                [32, 32, 16], activate_final=False),
            node_block_opt={"use_sent_edges": True},
            global_model_fn=lambda: snt.nets.MLP(
                [32, 16, 16, 4], activate_final=False))

    def __call__(self, graphs_data_dict, actions):
        graphs = gn.graphs.GraphsTuple(**graphs_data_dict)
        graphs = graphs._replace(globals=actions)

        encoded = self.graph_network(graphs).globals
        logits = encoded[:, 0]
        out = tf.sigmoid(logits)  # probability that jump action is correct

        fleets = encoded[:, 1:]  # amount of fleets of player 1

        return out, logits, fleets
