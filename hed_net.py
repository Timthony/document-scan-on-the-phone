import tensorflow as tf
import tensorflow.contrib.slim as slim
def hed_net(inputs, batch_size):
    with tf.variable_scope('hed', 'hed', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn = tf.nn.relu,
                            weights_initializer = tf.truncated_normal_initializer(0.0,0.01),
                            weights_regularizer = slim.l2_regularizer(0.0005)):
            # vgg16 conv && max_pool layers
            net = slim.repeat(inputs, 2, slim.conv2d, 12, [3, 3], scope='conv1')
            dsn1 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 24, [3, 3], scope='conv2')
            dsn2 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')





