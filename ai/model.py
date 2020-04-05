"""TensorFlow models"""
import graph_nets as gn
import sonnet as snt
import tensorflow as tf


class StateEncoder(snt.Module):
    """Module which encodes level graph into a vector"""

    def __init__(self):
        super().__init__(name="state_encoder")
        self.graph_network = gn.modules.GraphNetwork(
            edge_model_fn=lambda: snt.Linear(8),
            node_model_fn=lambda: snt.nets.MLP([32, 16]),
            node_block_opt={"use_sent_edges": True},
            global_model_fn=lambda: snt.nets.MLP(
                [32, 16, 1], activate_final=False))

    def __call__(self, graphs_data_dict, actions):
        graphs = gn.graphs.GraphsTuple(**graphs_data_dict)
        graphs = graphs._replace(globals=actions)

        logits = self.graph_network(graphs).globals
        out = tf.sigmoid(logits)
        return out, logits
