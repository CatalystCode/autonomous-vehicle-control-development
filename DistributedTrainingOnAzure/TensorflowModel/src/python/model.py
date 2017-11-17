# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

"""
Simple convolutional neural network to classify image.

Includes two convolutional layers, each followed by a max pooling layer. Finally,
there are two hidden dense layers before the final output.
"""
import tensorflow as tf

def inference(image_batch):

    input_layer=tf.reshape(image_batch,[-1,3,224,224])

    with tf.name_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=11, strides=4,
                                 use_bias=True, activation=tf.nn.relu, data_format='channels_first',
                                 padding='VALID', name='conv1')

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2,  data_format='channels_first',
                                padding='VALID', name='pool1')

    with tf.name_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=5, strides=1,
                                 use_bias=True, activation=tf.nn.relu, data_format='channels_first',
                                 padding='VALID', name='conv2')

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2, data_format='channels_first',
                                padding='VALID', name='pool2')

    with tf.name_scope('dense1') as scope:
        pool2_flat = tf.reshape(pool2, [-1,10*10*256])
        dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,
                                 trainable=True, use_bias=True, name='dense1')

    with tf.name_scope('dense2') as scope:
        dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu,
                                 trainable=True, use_bias=True, name='dense2')

    logits = tf.layers.dense(inputs=dense2, units=2, name='logits')

    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name="classes"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return logits, predictions

def loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(labels - 1), logits=logits, name='cross_entropy_single')
    return tf.reduce_mean(loss, name='cross_entropy_batch')

def accuracy(logits, labels):
    acc = tf.equal(tf.cast((labels - 1), tf.int64), tf.argmax(logits, 1))
    return tf.reduce_mean(tf.cast(acc, tf.float32), name='accuracy')