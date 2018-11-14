#coding=utf8
# 训练hed网络

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import tensorflow as tf

from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion('1.6'), '请使用Tensorflow version 1.6 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

import const
from util import *
from input_pipeline import *
from hed_net import *

from tensorflow import flags

