# Copyright 2016 Google Inc. All Rights Reserved.
# Modified 2017 xiou@microsoft.com. Significantly changed to support cars dataset, NCHW format,
# and new TF model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import image_processing
import model
from data import CarsData

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_integer('num_examples', 200,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")

tf.app.flags.DEFINE_string('subset', 'validation', 'Either "train" or "validation".')

def main(unused_args):
    dataset = CarsData(subset=FLAGS.subset)
    assert dataset.data_files()

    with tf.Graph().as_default():

        # Get images and labels from the dataset.
        images, labels = image_processing.batch_inputs(
            dataset=dataset,
            batch_size=FLAGS.batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            train=False,
            regular=True
        )

        images_NCHW = tf.transpose(images, (0,3,1,2))

        # Build a Graph that computes the logits predictions from the
        # inference model.
        print "Setting up model"
        logits, predictions = model.inference(images_NCHW)

        # Calculate predictions.
        top_1_op = tf.nn.in_top_k(logits, (labels - 1), 1)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                               graph_def=graph_def)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:

                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                                     ckpt.model_checkpoint_path))

                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/imagenet_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                print('Successfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))

            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                # Counts the number of correct predictions.
                count_top_1 = 0.0
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0

                print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
                start_time = time.time()
                while step < num_iter and not coord.should_stop():
                    top_1 = sess.run([top_1_op])

                    count_top_1 += np.sum(top_1)
                    step += 1
                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = FLAGS.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                              'sec/batch)' % (datetime.now(), step, num_iter,
                                              examples_per_sec, sec_per_batch))
                        start_time = time.time()

                # Compute precision @ 1
                precision_at_1 = count_top_1 / total_sample_count
                print('%s: precision @ 1 = %.4f [%d examples]' %
                      (datetime.now(), precision_at_1, total_sample_count))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
                summary_writer.add_summary(summary, global_step)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':
    tf.app.run()

