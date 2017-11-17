# Copyright 2016 Google Inc. All Rights Reserved.
# Modified 2017 xiou@microsoft.com. Significant modifications made to use 
# a MonitoredTrainingSession, new Tensorflow model, and NCHW format.
#
# Code based loosely on 
# https://github.com/tensorflow/models/blob/master/research/inception/inception/imagenet_distributed_train.py
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
import image_processing
import model
import numpy as np
import os
import tensorflow as tf

from data import CarsData

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')

tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Communication protocol to use in distributed """
                           """execution (default grpc) """)

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')

tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in SyncReplicasOptimizer.
tf.app.flags.DEFINE_integer('task_id', 0,
                            'Task ID of the worker/replica running the training.')

# More details can be found in the SyncReplicasOptimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py

tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")

tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')

tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

tf.app.flags.DEFINE_bool('force_use_cpu', False,
                         'Whether to force execution to CPU')

def main(unused_args):

    if FLAGS.force_use_cpu:
      os.environ['CUDA_VISIBLE_DEVICES'] = ''

    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    tf.logging.info('Worker hosts are: %s' % worker_hosts)

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})

    server = tf.train.Server(
        cluster_spec.as_dict(),
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_id,
        protocol=FLAGS.protocol)

    if FLAGS.job_name == 'ps':
        print "I'm a parameter server."
        server.join()

    else:
        dataset = CarsData(subset=FLAGS.subset)
        assert dataset.data_files()

        if FLAGS.task_id == 0:
            if not tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.MakeDirs(FLAGS.train_dir)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_id,
                cluster=cluster_spec
        )):
            print "I'm a worker"

            images_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(None,3,224,224),
                                                name="images_placeholder")

            labels_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=(None),
                                                name="labels_placeholder")

            images, labels = image_processing.batch_inputs(
                dataset=dataset,
                batch_size=FLAGS.batch_size,
                num_preprocess_threads=FLAGS.num_preprocess_threads,
                train=True,
                regular=True
            )

            logits, predictions = model.inference(images_placeholder)

            loss = model.loss(logits, labels_placeholder)
            accuracy = model.accuracy(logits, labels_placeholder)
            global_step = tf.contrib.framework.get_or_create_global_step()

            opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)

            num_workers = len(cluster_spec.as_dict()['worker'])
            print "Number of workers: %d" % num_workers

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=num_workers,
                total_num_replicas=num_workers,
                use_locking=True,
                name='sync_replicas')

            train_op = opt.minimize(loss=loss, global_step=global_step)

            is_chief = (FLAGS.task_id == 0)
            print "IsChief: %s" % is_chief

            sync_replicas_hook = opt.make_session_run_hook(
                is_chief=is_chief,
                num_tokens=num_workers)

            last_step_hook = tf.train.StopAtStepHook(num_steps=FLAGS.max_steps)

            hooks = [sync_replicas_hook, last_step_hook]

            with tf.train.MonitoredTrainingSession(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=FLAGS.log_device_placement
                ),
                master=server.target,
                is_chief=is_chief,
                checkpoint_dir=FLAGS.train_dir,
                hooks=hooks,
                stop_grace_period_secs=120) as mon_session:

                while not mon_session.should_stop():

                    image_feed, label_feed = mon_session.run([images,labels])

                    # Need to check mon_session.should_stop after each session.run to avoid
                    # errors when calling run after stop
                    if (mon_session.should_stop()):
                        break

                    # Convert from NHWC to NCHW format
                    image_feed_NCHW = np.transpose(image_feed, (0,3,1,2))

                    feed_dict = {
                        images_placeholder: image_feed_NCHW,
                        labels_placeholder: label_feed
                    }

                    print "==================================================="
                    print "Running train op"
                    _, current_loss, current_step, current_accuracy = \
                        mon_session.run([train_op, loss, global_step, accuracy],
                                        feed_dict = feed_dict)
                    print "Current step: %s" % current_step
                    print "Current loss: %.2f" % current_loss
                    print "Current accuracy: %.4f" % current_accuracy
                    print "==================================================="

if __name__ == '__main__':
    tf.app.run()