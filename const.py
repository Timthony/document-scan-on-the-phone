#coding=utf8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class _const:
    class ConstError(TypeError):
        pass
    def __setattr__(self, key, value):
        if self.__dict__.has_key(name):
            raise self.ConstError, "Can't rebind const(%s)"%name
        self.__dict__[name]=value
import sys
sys.modules[__name__] = _const

import const

import os

const.image_height = 256
const.image_width = 256


'''
如果使用 mobilenet_v2_style_hed 或 mobilenet_v1_style_hed，
一定要设置 const.use_batch_norm = True，因为 MobileNet 本身就要求使用 batch norm。
L2 regularizer 可用可不用，目前我选择的是不使用。
'''
const.use_batch_norm = True
const.use_kernel_regularizer = False





