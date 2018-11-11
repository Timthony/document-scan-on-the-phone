#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import const

import tensorflow as tf

def mobilenet_v2_func_blocks(is_training):
    assert const.use_batch_norm == True
    filter_initializer = tf.contrib.layers.xavier_initializer()
    activation_func = tf.nn.relu6

    def con2d(inputs, filters, kernel_size, stride, scope=''):
        with tf.variable_scope(scope):
            with tf.variable_scope('conv2d'):
                outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(stride, stride),
                                           padding='same', activation=None, use_bias=False,
                                           kernel_initializer=filter_initializer)
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)

            return outputs

    def _1x1_conv2d(inputs, filters, stride):
        kernel_size = [1, 1]
        with tf.variable_scope('1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(stride, stride),
                                       padding='same', activation=None,use_bias=False,
                                       kernel_initializer=filter_initializer)
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        return outputs

    def expansion_conv2d(inputs, expansion, stride):
        input_shape = inputs.get_shape().as_list()
        assert len(input_shape) == 4
        filters = input_shape[3] * expansion

        kernel_size = [1, 1]
        with tf.variable_scope('expansion_1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(stride, stride),
                                       padding='same', activation=None, use_bias=False,
                                       kernel_initializer=filter_initializer)
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = activation_func(outputs)
        return outputs

    def projection_conv2d(inputs, filters, stride):
        kernel_size = [1, 1]
        with tf.variable_scope('projection_1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(stride, stride),
                                       padding='same', activation=None, use_bias=False,
                                       kernel_initializer=filter_initializer)
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        return outputs








def mobilenet_v2(inuts, is_training):
    pass