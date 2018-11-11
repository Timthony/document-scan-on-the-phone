#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import const
from mobilenet import *

import tensorflow as tf



def class_balanced_sigmoid_cross_entropy(logits, label):
    """
    :param logits: of shape (b, ...).
    :param label:of the same shape. the ground truth in {0,1}.
    :return:class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        count_neg = tf.reduce_sum(1.0 - label)     # 样本中0的数量
        count_pos = tf.reduce_sum(label)           # 样本中1的数量，远小于count_neg
        beta = count_neg/(count_neg + count_pos)
        pos_weight = beta/(1.0-beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)

        # 如果样本中1的数量等于0，那就直接让 cost 为 0，因为 beta == 1 时， 除法 pos_weight = beta / (1.0 - beta) 的结果是无穷大
        zero = tf.equal(count_pos, 0.0)
        final_cost = tf.where(zero, 0.0, cost)
    return final_cost


def mobilenet_v2_style_hed(inputs, batch_size, is_training):
    assert const.use_batch_norm == True
    assert const.use_kernel_regularizer == False

    if const.use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    else:
        weights_regularizer = None

    func_blocks = mobilenet_v2_func_blocks(is_training)





