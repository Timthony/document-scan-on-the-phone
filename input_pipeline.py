#!/usr/bin/python
# coding=utf8
# 进度：已完成
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import const
from util import *

import tensorflow as tf

###############################################################################
#######     固定尺寸的 image，不需要 tf.image.resize，而是用 tf.reshape    ########
###############################################################################
# 读取固定尺寸的图像存为tensor
def read_fix_size_image_format(dataset_root_dir_string, filename_queue):
    # http://stackoverflow.com/questions/37198357/loading-images-and-labels-from-csv-file-using-tensorflow
    # https://www.tensorflow.org/versions/r0.11/how_tos/reading_data/index.html#file-formats
    reader = tf.TextLineReader()                 # 每次读取一行csv文件
    key, value = reader.read(filename_queue)

    # Default values
    record_defaults = [[''], ['']]
    # 使用tf.decode_csv来对每一行进行解析，分割为图像路径和标签路径
    image_path, annotation_path = tf.decode_csv(value, record_defaults=record_defaults)

    # csv 里保存的不是绝对路径，需要和 dataset_root_dir_string 一起拼装成完整的路径
    image_path = tf.string_join([tf.constant(dataset_root_dir_string), image_path])
    annotation_path = tf.string_join([tf.constant(dataset_root_dir_string), annotation_path])
    # 对最后的图像和标签路径进行读取
    image_content = tf.read_file(image_path)
    annotation_content = tf.read_file(annotation_path)

    # http://stackoverflow.com/questions/34746777/why-do-i-get-valueerror-image-must-be-fully-defined-when-transforming-im
    # http://stackoverflow.com/questions/37772329/tensorflow-tensor-set-shape-valueerror-image-must-be-fully-defined
    # image is jpg, annotation is png
    # 对jpg格式和png格式的图像进行解码，得到图像的像素值，
    # 这个像素值可以用于显示图像。如果没有解码，读取的图像是一个字符串
    image_tensor = tf.image.decode_jpeg(image_content, channels=3)
    annotation_tensor = tf.image.decode_png(annotation_content, channels=1)

    # decode之后，一定要设置 image 的大小，或者 resize 到一个size，否则会 crash
    image_tensor = tf.reshape(image_tensor, [const.image_height, const.image_width, 3])
    annotation_tensor = tf.reshape(annotation_tensor, [const.image_height, const.image_width, 1])

    image_float = tf.to_float(image_tensor)
    annotation_float = tf.to_float(annotation_tensor)
    # print('debug, image_float shape is: {}'.format(image_float.get_shape()))
    # print('debug, annotation_float shape is: {}'.format(annotation_float.get_shape()))

    if const.use_batch_norm == True:
        image_float = image_float / 255.0         # 进行归一化处理
    else:
        # 这个分支主要是为了匹配不使用 batch norm 时的 VGG
        image_float = mean_image_subtraction(image_float, [R_MEAN, G_MEAN, B_MEAN])  # 一个不做归一化，一个做归一化处理

    # 不管是不是 VGG，annotation 都需要归一化
    annotation_float = annotation_float / 255.0

    return image_float, annotation_float, image_path

# 打乱数据，进行训练前的操作
def fix_size_image_pipeline(dataset_root_dir_string, filename_queue, batch_size, num_epochs=None):

    image_tensor, annotation_tensor, image_path = read_fix_size_image_format(dataset_root_dir_string, filename_queue)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size
    # 将队列中数据打乱后，再读取出来
    image_tensor, annotation_tensor, image_path_tensor = tf.train.shuffle_batch(
        [image_tensor, annotation_tensor, image_path],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return image_tensor, annotation_tensor, image_path_tensor


