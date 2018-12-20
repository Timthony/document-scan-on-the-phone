#coding=utf8
# 定义hed网络
# 进度：已完成
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import const
from mobilenet import *

import tensorflow as tf


# 论文给出的 HED 网络是一个通用的边缘检测网络，
# 按照论文的描述，每一个尺度上得到的 image，都需要参与 cost 的计算
# 按照这种方式训练出来的网络，检测到的边缘线是有一点粗的，
# 为了得到更细的边缘线，通过多次试验找到了一种优化方案
# 也就是不再让每个尺度上得到的 image 都参与 cost 的计算，
# 只使用融合后得到的最终 image 来进行计算。
# 另外还有一点，按照 HED 论文里的要求，计算 cost 的时候，不能使用常见的方差 cost，
# 而应该使用 cost-sensitive loss function，
def class_balanced_sigmoid_cross_entropy(logits, label):
    """
    :param logits: of shape (b, ...).
    :param label:of the same shape. the ground truth in {0,1}.
    :return:class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        count_neg = tf.reduce_sum(1.0 - label)     # 样本中0的数量, 负样本
        count_pos = tf.reduce_sum(label)           # 样本中1的数量，表示边缘，边缘的像素点远小于count_neg，类别不平衡，所以不直接计算损失
        beta = count_neg/(count_neg + count_pos)
        pos_weight = beta/(1.0-beta)               # 大于1的值
		# tf.nn.weighted_cross_entropy_with_logits和sigmoid_cross_entropy_with_logits()相似，区别就是加入了pos_weight
		# 用来平衡查准率和查全率，在边缘检测中，边缘总是少数的，大部分都是非边缘，所以类别极不平衡
		# 计算方法targets * -log(sigmoid(logits)) * pos_weight +
        #     (1 - targets) * -log(1 - sigmoid(logits))
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1-beta))
        # 如果样本中1的数量等于0，那就直接让 cost 为 0，因为 beta == 1 时， 除法 pos_weight = beta / (1.0 - beta) 的结果是无穷大
        zero = tf.equal(count_pos, 0.0)
        final_cost = tf.where(zero, 0.0, cost)
		# Return the elements, either from x or y, depending on the condition.
	    # 如果zero为true，那么返回0.0， 如果zero为false，则返回cost
    return final_cost


def mobilenet_v2_style_hed(inputs, batch_size, is_training):
    assert const.use_batch_norm == True
    assert const.use_kernel_regularizer == False

    if const.use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    else:
        weights_regularizer = None

    func_blocks = mobilenet_v2_func_blocks(is_training)

    _conv2d = func_blocks['conv2d']
    _inverted_residual_block = func_blocks['inverted_residual_block']
    _avg_pool2d = func_blocks['avg_pool2d']
    filter_initializer = func_blocks['filter_initializer']
    activation_func = func_blocks['activation_func']
    ####################################################

    def _dsn_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs, filters, kernel_size,
                                   padding='same',
                                   activation=None,
                                   use_bias=False,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)

        return outputs

    def _output_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs, filters, kernel_size,
                                   padding='same',
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        return outputs

    def _dsn_deconv2d_with_upsample_factor(inputs, filters, upsample_factor):
        ## https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
        ## 从这个原版代码里看，是这样计算 kernel_size 的
        kernel_size = [2 * upsample_factor, 2 * upsample_factor]
        outputs = tf.layers.conv2d_transpose(inputs,
                                             filters,
                                             kernel_size,
                                             strides=(upsample_factor, upsample_factor),
                                             padding='same',
                                             activation=None,  ## no activation
                                             use_bias=True,  ## use bias
                                             kernel_initializer=filter_initializer,
                                             kernel_regularizer=weights_regularizer)

        ## 概念上来说，deconv2d 已经是最后的输出 layer 了，只不过最后还有一步 1x1 的 conv2d 把 5 个 deconv2d 的输出再融合到一起
        ## 所以不需要再使用 batch normalization 了

        return outputs

    with tf.variable_scope('hed', 'hed', [inputs]):
        end_points = {}
        net = inputs

        # mobilenet v2 as basenet
        with tf.variable_scope('mobilenet_v2'):
        # 标准的 mobilenet v2 里面并没有这两层，
        # 这里是为了得到和 input image 相同 size 的 feature map 而增加的层
            net = _conv2d(net, 3, [3, 3], stride=1, scope='block0_0')
            net = _conv2d(net, 6, [3, 3], stride=1, scope='block0_1')

            dsn1 = net
            net = _conv2d(net, 12, [3, 3], stride=2, scope='block0_2') # size/2

            net = _inverted_residual_block(net, 6, stride=1, expansion=1, scope='block1_0')

            dsn2 = net
            net = _inverted_residual_block(net, 12, stride=2, scope='block2_0')  # size/4
            net = _inverted_residual_block(net, 12, stride=1, scope='block2_1')

            dsn3 = net
            net = _inverted_residual_block(net, 24, stride=2, scope='block3_0')  # size/8
            net = _inverted_residual_block(net, 24, stride=1, scope='block3_1')
            net = _inverted_residual_block(net, 24, stride=1, scope='block3_2')

            dsn4 = net
            net = _inverted_residual_block(net, 48, stride=2, scope='block4_0')  # size/16
            net = _inverted_residual_block(net, 48, stride=1, scope='block4_1')
            net = _inverted_residual_block(net, 48, stride=1, scope='block4_2')
            net = _inverted_residual_block(net, 48, stride=1, scope='block4_3')

            net = _inverted_residual_block(net, 64, stride=1, scope='block5_0')
            net = _inverted_residual_block(net, 64, stride=1, scope='block5_1')
            net = _inverted_residual_block(net, 64, stride=1, scope='block5_2')

            dsn5 = net

        ## dsn layers
        with tf.variable_scope('dsn1'):
            dsn1 = _dsn_1x1_conv2d(dsn1, 1)
            # print('!! debug, dsn1 shape is: {}'.format(dsn1.get_shape()))
            ## no need deconv2d

        with tf.variable_scope('dsn2'):
            dsn2 = _dsn_1x1_conv2d(dsn2, 1)
            # print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
            dsn2 = _dsn_deconv2d_with_upsample_factor(dsn2, 1, upsample_factor=2)
            # print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))

        with tf.variable_scope('dsn3'):
            dsn3 = _dsn_1x1_conv2d(dsn3, 1)
            # print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))
            dsn3 = _dsn_deconv2d_with_upsample_factor(dsn3, 1, upsample_factor=4)
            # print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))

        with tf.variable_scope('dsn4'):
            dsn4 = _dsn_1x1_conv2d(dsn4, 1)
            # print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))
            dsn4 = _dsn_deconv2d_with_upsample_factor(dsn4, 1, upsample_factor=8)
            # print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))

        with tf.variable_scope('dsn5'):
            dsn5 = _dsn_1x1_conv2d(dsn5, 1)
            # print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
            dsn5 = _dsn_deconv2d_with_upsample_factor(dsn5, 1, upsample_factor=16)
            # print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))

        # dsn fuse
        with tf.variable_scope('dsn_fuse'):
            dsn_fuse = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], 3)
            # print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))
            dsn_fuse = _output_1x1_conv2d(dsn_fuse, 1)
            # print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5




def mobilenet_v1_style_hed(inputs, batch_size, is_training):
    # 前面一部分就是定义实现不同功能的各种 layer，
    # 后面部分就是用各种 layer 来组装 net 的主体结构。
    assert const.use_batch_norm == True
    assert const.use_kernel_regularizer == False

    alpha = 1.0
    filter_initializer = tf.contrib.layers.xavier_initializer()

    if const.use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
    else:
        weights_regularizer = None

    def _conv2d(inputs, filters, kernel_size, stride, scope=''):
        with tf.variable_scope(scope):
            outputs = tf.layers.conv2d(inputs,
                                       filters,
                                       kernel_size,
                                       strides=(stride, stride),
                                       padding='same',
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=filter_initializer)
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = tf.nn.relu(outputs)
        return outputs

    '''stride is just for tf.layers.separable_conv2d, means depthwise_conv_stride'''
    def _depthwise_conv2d(inputs,
                          pointwise_conv_filters,
                          depthwise_conv_kernel_size,
                          stride,
                          scope=''):
        with tf.variable_scope(scope):
            with tf.variable_scope('depthwise_conv'):
                outputs = tf.contrib.layers.separable_conv2d(
                            inputs,
                            None, # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
                            depthwise_conv_kernel_size,
                            depth_multiplier=1,
                            stride=(stride, stride),
                            padding='SAME',
                            activation_fn=None,
                            weights_initializer=filter_initializer,
                            biases_initializer=None)
                '''
                !!!important!!! tf.contrib.layers.separable_conv2d already has a depthwise convolution and a pointwise convolution,
                but By passing num_outputs=None, separable_conv2d produces only a depthwise convolution layer 
                ref -- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py 
                '''

            with tf.variable_scope('pointwise_conv'):
                pointwise_conv_filters = int(pointwise_conv_filters * alpha)
                outputs = tf.layers.conv2d(outputs,
                                        pointwise_conv_filters, ##!! here, pointwise_conv_filters * alpha
                                        (1, 1),
                                        padding='same',
                                        activation=None,
                                        use_bias=False,
                                        kernel_initializer=filter_initializer)

                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)

        return outputs

    def _dsn_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size,
                                   padding='same',
                                   activation=None,  ## no activation
                                   use_bias=False,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        ## no activation

        return outputs

    def _output_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size,
                                   padding='same',
                                   activation=None,  ## no activation
                                   use_bias=True,  ## use bias
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        ## no batch normalization
        ## no activation

        return outputs

    def _dsn_deconv2d_with_upsample_factor(inputs, filters, upsample_factor):
        ## https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
        ## 从这个原版代码里看，是这样计算 kernel_size 的
        kernel_size = [2 * upsample_factor, 2 * upsample_factor]
        outputs = tf.layers.conv2d_transpose(inputs,
                                             filters,
                                             kernel_size,
                                             strides=(upsample_factor, upsample_factor),
                                             padding='same',
                                             activation=None,  ## no activation
                                             use_bias=True,  ## use bias
                                             kernel_initializer=filter_initializer,
                                             kernel_regularizer=weights_regularizer)

        ## 概念上来说，deconv2d 已经是最后的输出 layer 了，只不过最后还有一步 1x1 的 conv2d 把 5 个 deconv2d 的输出再融合到一起
        ## 所以不需要再使用 batch normalization 了

        return outputs

    with tf.variable_scope('hed', 'hed', [inputs]):
        end_points = {}
        net = inputs

        ## mobilenet v1 as base net
        with tf.variable_scope('mobilenet_v1'):
            # 标准的 mobilenet v1 里面并没有这两层，
            # 这里是为了得到和 input image 相同 size 的 feature map 而增加的层
            net = _conv2d(net, 6, [3, 3], stride=1, scope='extra_block0')
            net = _conv2d(net, 6, [3, 3], stride=1, scope='extra_block1')

            dsn1 = net
            net = _conv2d(net, 8, [3, 3], stride=2, scope='block0')
            # print('\r ++++ block0 shape: %s' % (net.get_shape().as_list()))
            end_points['block0'] = net
            net = _depthwise_conv2d(net, 16, [3, 3], stride=1, scope='block1')
            end_points['block1'] = net

            dsn2 = net
            net = _depthwise_conv2d(net, 32, [3, 3], stride=2, scope='block2')
            end_points['block2'] = net
            net = _depthwise_conv2d(net, 32, [3, 3], stride=1, scope='block3')
            end_points['block3'] = net

            dsn3 = net
            net = _depthwise_conv2d(net, 64, [3, 3], stride=2, scope='block4')
            end_points['block4'] = net
            net = _depthwise_conv2d(net, 64, [3, 3], stride=1, scope='block5')
            end_points['block5'] = net

            dsn4 = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=2, scope='block6')
            end_points['block6'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block7')
            end_points['block7'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block8')
            end_points['block8'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block9')
            end_points['block9'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block10')
            end_points['block10'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block11')
            end_points['block11'] = net

            dsn5 = net

        ## dsn layers
        with tf.variable_scope('dsn1'):
            dsn1 = _dsn_1x1_conv2d(dsn1, 1)
            print('!! debug, dsn1 shape is: {}'.format(dsn1.get_shape()))
            ## no need deconv2d

        with tf.variable_scope('dsn2'):
            dsn2 = _dsn_1x1_conv2d(dsn2, 1)
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
            dsn2 = _dsn_deconv2d_with_upsample_factor(dsn2, 1, upsample_factor=2)
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))

        with tf.variable_scope('dsn3'):
            dsn3 = _dsn_1x1_conv2d(dsn3, 1)
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))
            dsn3 = _dsn_deconv2d_with_upsample_factor(dsn3, 1, upsample_factor=4)
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))

        with tf.variable_scope('dsn4'):
            dsn4 = _dsn_1x1_conv2d(dsn4, 1)
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))
            dsn4 = _dsn_deconv2d_with_upsample_factor(dsn4, 1, upsample_factor=8)
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))

        with tf.variable_scope('dsn5'):
            dsn5 = _dsn_1x1_conv2d(dsn5, 1)
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
            dsn5 = _dsn_deconv2d_with_upsample_factor(dsn5, 1, upsample_factor=16)
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))

        # dsn fuse
        with tf.variable_scope('dsn_fuse'):
            dsn_fuse = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], 3)
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))
            dsn_fuse = _output_1x1_conv2d(dsn_fuse, 1)
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5


# 原始的HED网络，使用了裁剪版的vgg16，最后返回融合结果和五个边路结果
def vgg_style_hed(inputs, batch_size, is_training):
	# 通过使用这种初始化方法，我们能够保证输入变量的变化尺度不变，从而避免变化尺度在最后一层网络中爆炸或者弥散。
    filter_initializer = tf.contrib.layers.xavier_initializer()
    if const.use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
    else:
        weights_regularizer = None
    # 定义vgg网络的通用结构，方便后边直接调用
    def _vgg_conv2d(inputs, filters, kernel_size):
        use_bias = True
        if const.use_batch_norm:
            use_bias = False

        outputs = tf.layers.conv2d(inputs, filters, kernel_size,
                                   padding='same',activation=None,
                                   use_bias=use_bias,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        if const.use_batch_norm:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = tf.nn.relu(outputs)
        return outputs

    def _max_pool2d(inputs):
        outputs = tf.layers.max_pooling2d(inputs, [2, 2], strides=(2, 2), padding='same')
        return outputs

    def _dsn_1x1_conv2d(inputs, filters):
        use_bias = True
        if const.use_batch_norm:
            use_bias = False
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs, filters,kernel_size,
                                   padding='same', activation=None,
                                   use_bias=use_bias,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        if const.use_batch_norm:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)

        return outputs

    def _output_1x1_conv2d(inputs, filters):
        kernel_size = [1,1]
        outputs = tf.layers.conv2d(inputs, filters, kernel_size, padding='same',
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        ## 不用batch normalization
        ## 不同激活

        return outputs
    # 反卷积，将图像变为原始图大小，方便融合
    def _dsn_deconv2d_with_upsample_factor(inputs, filters, upsample_factor):
        kernel_size = [2 * upsample_factor, 2 * upsample_factor]
        outputs = tf.layers.conv2d_transpose(inputs, filters, kernel_size,
                                             strides=(upsample_factor, upsample_factor),
                                             padding='same',
                                             activation=None,
                                             use_bias=True,
                                             kernel_initializer=filter_initializer,
                                             kernel_regularizer=weights_regularizer)
        # 概念上来说，deconv2d 已经是最后的输出 layer 了，
        # 只不过最后还有一步 1x1 的 conv2d 把 5 个 deconv2d 的输出再融合到一起
        # 所以不需要再使用 batch normalization 了
        return outputs
    # 定义HED网络，利用上边定义的各种layer，结构参照了vgg16
    with tf.variable_scope('hed', 'hed', [inputs]):
        end_points = {}
        net = inputs

        with tf.variable_scope('conv1'):
            net = _vgg_conv2d(net, 12, [3, 3])
            net = _vgg_conv2d(net, 12, [3, 3])
            dsn1 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv2'):
            net = _vgg_conv2d(net, 24, [3, 3])
            net = _vgg_conv2d(net, 24, [3, 3])
            dsn2 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv3'):
            net = _vgg_conv2d(net, 48, [3, 3])
            net = _vgg_conv2d(net, 48, [3, 3])
            net = _vgg_conv2d(net, 48, [3, 3])
            dsn3 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv4'):
            net = _vgg_conv2d(net, 96, [3, 3])
            net = _vgg_conv2d(net, 96, [3, 3])
            net = _vgg_conv2d(net, 96, [3, 3])
            dsn4 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv5'):
            net = _vgg_conv2d(net, 192, [3, 3])
            net = _vgg_conv2d(net, 192, [3, 3])
            net = _vgg_conv2d(net, 192, [3, 3])
            dsn5 = net
            # 此处不需要池化

        #######【dsn layers边路层，HED的边路预测边缘的层，一共有五层和最后的融合层】######
        with tf.variable_scope('dsn1'):
            dsn1 = _dsn_1x1_conv2d(dsn1, 1)
            print('!! debug, dsn1 shape is: {}'.format(dsn1.get_shape()))
            ## no need deconv2d，因为这一层的输出与输入一样

        with tf.variable_scope('dsn2'):
            dsn2 = _dsn_1x1_conv2d(dsn2, 1)
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
            dsn2 = _dsn_deconv2d_with_upsample_factor(dsn2, 1, upsample_factor=2) # 放大2倍
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
        with tf.variable_scope('dsn3'):
            dsn3 = _dsn_1x1_conv2d(dsn3, 1)
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))
            dsn3 = _dsn_deconv2d_with_upsample_factor(dsn3, 1, upsample_factor=4) # 放大4倍
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))

        with tf.variable_scope('dsn4'):
            dsn4 = _dsn_1x1_conv2d(dsn4, 1)
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))
            dsn4 = _dsn_deconv2d_with_upsample_factor(dsn4, 1, upsample_factor=8) # 放大8倍
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))

        with tf.variable_scope('dsn5'):
            dsn5 = _dsn_1x1_conv2d(dsn5, 1)
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
            dsn5 = _dsn_deconv2d_with_upsample_factor(dsn5, 1, upsample_factor=16) # 放大16倍
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
        #################【将5个边路层融合】###################
        with tf.variable_scope('dsn_fuse'):
            dsn_fuse = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], 3)   # 连接张量
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))
            dsn_fuse = _output_1x1_conv2d(dsn_fuse, 1)
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5








































































































