#coding=utf-8
# 定义mobilenet_v1和mobilenet_v2
# 进度：已完成
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Python提供了__future__模块，把下一个新版本的特性导入到当前版本，
# 于是我们就可以在当前版本中测试一些新版本的特性。
import sys
import os
import const

import tensorflow as tf

# mobilenet_v1网络定义
def mobilenet_v1(inputs, alpha, is_training):
    assert const.use_batch_norm == True
    # assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假
    # 缩小因子， 只能为1，0.75，0.5，0.25
    if alpha not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError('alpha can be one of'
                         '`0.25`, `0.50`, `0.75` or `1.0` only.')
    filter_initializer = tf.contrib.layers.xavier_initializer()
    # 卷积，BN，RELU
    def _conv2d(inputs, filters, kernel_size, stride, scope=''):
        with tf.variable_scope(scope):
            outputs = tf.layers.conv2d(inputs, filters, kernel_size,
                                       strides=(stride, stride), padding='same',
                                       activation=None, use_bias=False,
                                       kernel_initializer=filter_initializer)
            # 非线性激活之前进行BN批标准化
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = tf.nn.relu(outputs)
        return outputs
    # 深度可分离卷积，标准卷积分解成深度卷积(depthwise convolution)和逐点卷积(pointwise convolution)
    def _depthwise_conv2d(inputs,
                          pointwise_conv_filters,
                          depthwise_conv_kernel_size,
                          stride,
                          scope=''):
        with tf.variable_scope(scope):
            with tf.variable_scope('depthwise_conv'):
                outputs = tf.contrib.layers.separable_conv2d(
                    inputs,
                    None,
                    depthwise_conv_kernel_size,
                    depth_multiplier=1,
                    stride=(stride, stride),
                    padding='SAME',
                    activation_fn=None,
                    weights_initializer=filter_initializer,
                    biases_initializer=None)
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)
            with tf.variable_scope('pointwise_conv'):
                pointwise_conv_filters = int(pointwise_conv_filters * alpha)
                outputs = tf.layers.conv2d(outputs,
                                           pointwise_conv_filters,
                                           (1,1),
                                           padding='same',
                                           activation=None,
                                           use_bias=False,
                                           kernel_initializer=filter_initializer)
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)

        return outputs
    # 平均池化
    def _avg_pool2d(inputs, scope=''):
        inputs_shape = inputs.get_shape().as_list()
        assert len(inputs_shape) == 4

        pool_height = inputs_shape[1]
        pool_width  = inputs_shape[2]

        with tf.variable_scope(scope):
            outputs = tf.layers.average_pooling2d(inputs,
                                                  [pool_height, pool_width],
                                                  strides=(1, 1),
                                                  padding='valid')
        return outputs

    '''
        执行分类任务的网络结构，通常还可以作为实现其他任务的网络结构的 base architecture，
        为了方便代码复用，这里只需要实现出卷积层构成的主体部分，
        外部调用者根据各自的需求使用这里返回的 output 和 end_points。
        比如对于分类任务，按照如下方式使用这个函数

        image_height = 224
        image_width = 224
        image_channels = 3

        x = tf.placeholder(tf.float32, [None, image_height, image_width, image_channels])
        is_training = tf.placeholder(tf.bool, name='is_training')

        output, net = mobilenet_v1(x, 1.0, is_training)
        print('output shape is: %r' % (output.get_shape().as_list()))

        output = tf.layers.flatten(output)
        output = tf.layers.dense(output,
                            units=1024, # 1024 class
                            activation=None,
                            use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('output shape is: %r' % (output.get_shape().as_list()))
        '''
    with tf.variable_scope('mobilenet_v1', 'mobilenet_v1', [inputs]):
        end_points = {}
        net = inputs

        net = _conv2d(net, 32, [3,3], stride=2, scope='block0')
        end_points['block0'] = net
        net = _depthwise_conv2d(net, 64, [3, 3], stride=1, scope='block1')
        end_points['block1'] = net

        net = _depthwise_conv2d(net, 128, [3, 3], stride=2, scope='block2')
        end_points['block2'] = net
        net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block3')
        end_points['block3'] = net

        net = _depthwise_conv2d(net, 256, [3, 3], stride=2, scope='block4')
        end_points['block4'] = net
        net = _depthwise_conv2d(net, 256, [3, 3], stride=1, scope='block5')
        end_points['block5'] = net

        net = _depthwise_conv2d(net, 512, [3, 3], stride=2, scope='block6')
        end_points['block6'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block7')
        end_points['block7'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block8')
        end_points['block8'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block9')
        end_points['block9'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block10')
        end_points['block10'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block11')
        end_points['block11'] = net

        net = _depthwise_conv2d(net, 1024, [3, 3], stride=2, scope='block12')
        end_points['block12'] = net
        net = _depthwise_conv2d(net, 1024, [3, 3], stride=1, scope='block13')
        end_points['block13'] = net

        output = _avg_pool2d(net, scope='output')
    return output, end_points




# mobilenet_v2网络定义
def mobilenet_v2_func_blocks(is_training):
    assert const.use_batch_norm == True
    filter_initializer = tf.contrib.layers.xavier_initializer()
    activation_func = tf.nn.relu6
    # 普通卷积
    def conv2d(inputs, filters, kernel_size, stride, scope=''):
        with tf.variable_scope(scope):
            with tf.variable_scope('conv2d'):
                outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(stride, stride),
                                           padding='same', activation=None, use_bias=False,
                                           kernel_initializer=filter_initializer)
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)

            return outputs

    # 逐点卷积
    def _1x1_conv2d(inputs, filters, stride):
        kernel_size = [1, 1]
        with tf.variable_scope('1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(stride, stride),
                                       padding='same', activation=None,use_bias=False,
                                       kernel_initializer=filter_initializer)
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        return outputs
    # 升维
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
    # 降维
    def projection_conv2d(inputs, filters, stride):
        kernel_size = [1, 1]
        with tf.variable_scope('projection_1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=(stride, stride),
                                       padding='same', activation=None, use_bias=False,
                                       kernel_initializer=filter_initializer)
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        return outputs
    # 深度可分离卷积
    def depthwise_conv2d(inputs, depthwise_conv_kernel_size,stride):
        with tf.variable_scope('depthwise_conv2d'):
            outputs = tf.contrib.layers.separable_conv2d(
                inputs,
                None,
                depthwise_conv_kernel_size,
                depth_multiplier=1,
                stride=(stride,stride),
                padding='SAME',
                activation_fn=None,
                weights_initializer=filter_initializer,
                biases_initializer=None)
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = activation_func(outputs)
        return outputs
    #  池化
    def avg_pool2d(inputs, scope=''):
        inputs_shape = inputs.get_shape().as_list()
        assert len(inputs_shape) == 4

        pool_height = inputs_shape[1]
        pool_width = inputs_shape[2]

        with tf.variable_scope(scope):
            outputs = tf.layers.average_pooling2d(inputs, [pool_height, pool_width],
                                                  strides=(1, 1),padding='valid')

        return outputs
    # 倒置的残差
    def inverted_residual_block(inputs, filters, stride, expansion=6,scope=''):
        assert stride == 1 or stride == 2
        depthwise_conv_kernel_size = [3, 3]
        pointwise_conv_filters = filters

        with tf.variable_scope(scope):
            net = inputs
            net = expansion_conv2d(net, expansion, stride=1)  # 升维
            net = depthwise_conv2d(net, depthwise_conv_kernel_size, stride=stride) # 特征提取
            net = projection_conv2d(net, pointwise_conv_filters, stride=1) # 降维

            if stride == 1:
                # print('----------------- test, net.get_shape().as_list()[3] = %r' % net.get_shape().as_list()[3])
                # print('----------------- test, inputs.get_shape().as_list()[3] = %r' % inputs.get_shape().as_list()[3])
                # 如果 net.get_shape().as_list()[3] != inputs.get_shape().as_list()[3]
                # 借助一个 1x1 的卷积让他们的 channels 相等，然后再相加
                if net.get_shape().as_list()[3] != inputs.get_shape().as_list()[3]:
                    inputs = _1x1_conv2d(inputs, net.get_shape().as_list()[3], stride=1)

                net = net + inputs
                return net
            else:
                # stride == 2
                return net
    # 定义功能块的集合
    func_blocks = {}
    func_blocks['conv2d'] = conv2d
    func_blocks['inverted_residual_block'] = inverted_residual_block
    func_blocks['avg_pool2d'] = avg_pool2d
    func_blocks['filter_initializer'] = filter_initializer
    func_blocks['activation_func'] = activation_func

    return func_blocks



def mobilenet_v2(inputs, is_training):
    assert const.use_batch_norm == True

    func_blocks = mobilenet_v2_func_blocks(is_training)
    _conv2d = func_blocks['conv2d']
    _inverted_residual_block = func_blocks['inverted_residual_block']
    _avg_pool2d = func_blocks['avg_pool2d']

    with tf.variable_scope('mobilenet_v2', 'mobilenet_v2', [inputs]):
        end_points = {}
        net = inputs

        net = _conv2d(net, 32, [3, 3], stride=2, scope='block0_0')
        end_points['block0'] = net
        print('!! debug block0, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 16, stride=1, expansion=1, scope='block1_0')
        end_points['block1'] = net
        print('!! debug block1, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 24, stride=2, scope='block2_0')
        net = _inverted_residual_block(net, 24, stride=1, scope='block2_1')
        end_points['block2'] = net
        print('!! debug block2, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 32, stride=2, scope='block3_0')
        net = _inverted_residual_block(net, 32, stride=1, scope='block3_1')
        net = _inverted_residual_block(net, 32, stride=1, scope='block3_2')
        end_points['block3'] = net
        print('!! debug block3, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 64, stride=2, scope='block4_0')
        net = _inverted_residual_block(net, 64, stride=1, scope='block4_1')
        net = _inverted_residual_block(net, 64, stride=1, scope='block4_2')
        net = _inverted_residual_block(net, 64, stride=1, scope='block4_3')
        end_points['block4'] = net
        print('!! debug block4, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 96, stride=1, scope='block5_0')
        net = _inverted_residual_block(net, 96, stride=1, scope='block5_1')
        net = _inverted_residual_block(net, 96, stride=1, scope='block5_2')
        end_points['block5'] = net
        print('!! debug block5, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 160, stride=2, scope='block6_0')
        net = _inverted_residual_block(net, 160, stride=1, scope='block6_1')
        net = _inverted_residual_block(net, 160, stride=1, scope='block6_2')
        end_points['block6'] = net
        print('!! debug block6, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 320, stride=1, scope='block7_0')
        end_points['block7'] = net
        print('!! debug block7, net shape is: {}'.format(net.get_shape()))

        net = _conv2d(net, 1280, [1, 1], stride=1, scope='block8_0')
        end_points['block8'] = net
        print('!! debug block8, net shape is: {}'.format(net.get_shape()))

        output = _avg_pool2d(net, scope='output')
        print('!! debug after avg_pool, net shape is: {}'.format(output.get_shape()))

    return output, end_points