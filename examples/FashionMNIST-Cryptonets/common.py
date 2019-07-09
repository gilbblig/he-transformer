# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
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
# *****************************************************************************

import tensorflow as tf
import numpy as np
# import ngraph_bridge


def conv2d_stride_2_valid(x, W, name=None):
    """returns a 2d convolution layer with stride 2, valid pooling"""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def avg_pool_3x3_same_size(x):
    """3x3 avg_pool using same padding, keeping original feature map size"""
    return tf.nn.avg_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def get_variable(name, shape, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    if mode == 'train':
        return tf.get_variable(name, shape)
    else:
        return tf.constant(
            np.loadtxt(name + '.txt', dtype=np.float32).reshape(shape))


def cryptonets_model(x, mode):
    """Builds the graph for classifying digits based on Cryptonets

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is
        the number of pixels in a standard MNIST image.

    Returns:
        A tuple (y, a scalar placeholder). y is a tensor of shape
        (N_examples, 10), with values equal to the logits of classifying the
        digit into one of 10 classes (the digits 0-9).
    """
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()


    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]],
                               name='pad_const')
        x_image = tf.pad(x_image, paddings)


    with tf.name_scope('conv1'):
        # h_conv1 = tf.layers.conv2d(x_image, 5, 1, 2, 'valid', name=mode)
        W_conv1 = get_variable("W_conv1", [5, 5, 1, 5], mode)
        h_conv1 = tf.square(conv2d_stride_2_valid(x_image, W_conv1))
        
    # with tf.name_scope('pool1'):
    #     h_conv1 = avg_pool_3x3_same_size(h_conv1)


    # with tf.name_scope('conv2'):
    #     W_conv2 = get_variable("W_conv2", [5, 5, 5, 10], mode)
    #     h_conv2 = conv2d_stride_2_valid(h_conv1, W_conv2)
    # with tf.name_scope('fc1'):
    #     W_fc1 = get_variable("W_fc1", [5 * 5 * 10, 100], mode)
    #     h_conv2_flat = tf.reshape(h_conv2, [-1, 5 * 5 * 10])
    # h_full = tf.square(tf.matmul(h_conv2_flat, W_fc1))


    with tf.name_scope('fc1'):
        W_squash = get_variable('W_squash', [5 * 13 * 13, 100], mode)
        h_flat = tf.reshape(h_conv1, [-1, 5 * 13 * 13])
        # W_squash = get_variable('W_squash', [5 * 14 * 14, 100], mode)
        # h_flat = tf.reshape(h_conv1, [-1, 5 * 14 * 14])
    h_full = tf.square(tf.matmul(h_flat, W_squash))


    W_fc2 = get_variable('W_fc2', [100, 10], mode)
    y_conv = tf.matmul(h_full, W_fc2, name='logits_out')

    return y_conv
