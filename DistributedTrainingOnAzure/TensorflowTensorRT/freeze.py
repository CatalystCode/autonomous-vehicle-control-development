# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
"""
Based on code from Anthony.Turner@microsoft.com. Modified to remove training nodes.

Freezes a tensorflow graph to a .pb file from a given checkpoint folder. The 
frozen_model.pb file is written to the checkpoint folder.

Usage:
python freeze.py \
    --checkpoint_folder=/path/to/checkpoint_folder \
    --output_node_names=softmax_tensor
"""
import argparse
import sys

import tensorflow as tf
from tensorflow.python.framework import graph_util

parser = argparse.ArgumentParser(
    description='Freezes a tensorflow model to a .pb file.')

parser.add_argument('--checkpoint_folder',
                    type=str,
                    required=True,
                    help='Checkpoint folder to export')

parser.add_argument('--output_node_names',
                    type=str,
                    help='Output node[s] (y_pred or similar).')

def freeze_graph(model_folder, output_node_names):
    """Takes a model folder and creates a frozen model file."""
	
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path

        absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
        print "absolute_model_folder: %s" % absolute_model_folder

        output_graph = absolute_model_folder + "/frozen_model.pb"
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        with tf.Session() as sess:
            saver.restore(sess, input_checkpoint)
            output_graph_def = graph_util.convert_variables_to_constants(
              sess=sess,
              input_graph_def=input_graph_def,
              output_node_names=output_node_names.split(",")
            )

            output_graph_def = graph_util.remove_training_nodes(output_graph_def)

            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

    except:
        e = sys.exc_info()[0]
        print e

if __name__ == '__main__':

    args = parser.parse_args()

    freeze_graph(args.checkpoint_folder, args.output_node_names)