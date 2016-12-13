'''
Created on Nov 29, 2016

@author: george
'''
import tensorflow as tf
import numpy as np
from common import logger
from common.diskutils import ensure_dir
from skimage import io
import time
import os
import net
import os
import sys
import numpy.ma as ma
import tensorflow as tf

from tflearn import utils
from tflearn.config import *
from tflearn.metrics import *
from tflearn.data_utils import *
from tflearn.models.dnn import *
from tflearn.optimizers import *
from tflearn.activations import *
from tflearn.layers.conv import *
from tflearn.layers.core import *
from tflearn.helpers.trainer import *
from tflearn.initializations import *
from tflearn.layers.estimator import *
from tflearn.layers.normalization import *
from tensorflow import reshape as reshape
from tensorflow import squeeze as squeeze
from tensorflow import clip_by_value as clip
from tensorflow import transpose as transpose
from models.resnet_utils import batch_norm2
from _operator import is_

slim =tf.contrib.slim

def segnet_arg_scope(weight_decay=0.0001):
    """Defines the VGG arg scope.
    
    Args:
        weight_decay: The l2 regularization coefficient.
    
    Returns:
        An arg_scope.
    """
    batch_norm_decay=0.997
    batch_norm_epsilon=1e-5,
    batch_norm_scale=True
    
    batch_norm_params = {
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections':  None#tf.GraphKeys.UPDATE_OPS,
    }
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer):
        with slim.arg_scope([slim.conv2d], padding='SAME',   
                            normalizer_fn=batch_norm2,
                            normalizer_params=batch_norm_params) as arg_sc:
            return arg_sc
  

def poolargmax(name, tensor, kernel_size, strides=None, padding="SAME"):    
    return tf.nn.max_pool_with_argmax(    input     = tensor,
                                        ksize     = [1, kernel_size, kernel_size, 1],
                                        strides = [1, kernel_size, kernel_size, 1],
                                        padding = padding,
                                        Targmax = tf.int64,
                                        name     = name)


def unpoolind(x, shape, unpool_mat=None):
    """
    https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py
    Unpool the input with a fixed mat to perform kronecker product with.
    :param input: NHWC tensor
    :param shape: int or [h, w]
    :param unpool_mat: a tf/np matrix with size=shape. If None, will use a mat
        with 1 at top-left corner.
    :returns: NHWC tensor
    """
    input_shape = tf.shape(x)
    mat = np.zeros(shape, dtype='float32')
    mat[0][0] = 1
    unpool_mat = tf.constant(mat, name='unpool_mat')

    # perform a tensor-matrix kronecker product
    fx        = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1])
    fx        = tf.expand_dims(fx, -1)       # (bchw)x1
    mat        = tf.expand_dims(tf.reshape(unpool_mat, [-1]), 0)    #1x(shxsw)
    prod    = tf.matmul(fx, mat)    #(bchw) x(shxsw)
    prod    = tf.reshape(prod, tf.pack(
                [-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1]]))
    prod    = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod    = tf.reshape(prod, tf.pack(
                [-1, input_shape[1] * shape[0], input_shape[2] * shape[1], input_shape[3]]))
    return prod
#####

def dropout(name, tensor, keep_prob):
    return tf.nn.dropout(    x                    = tensor,
                            keep_prob            = keep_prob,
                            noise_shape            = None,
                            seed                = None,
                            name                = name)
class SegNet(object):
    '''
    classdocs
    '''


    def __init__(self,inp ,is_training):
        '''
        Constructor
        '''
        self.inp = inp
        self.is_training_pl = is_training
        basemodel_arg_scope = segnet_arg_scope()         
        with slim.arg_scope(basemodel_arg_scope):
                self.build()
        
    def build(self):
        
        with tf.variable_scope('vgg_16') as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(self.inp, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net, argpool1    = poolargmax    (name="pool1"  , tensor=net, kernel_size=2)
                
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net, argpool2    = poolargmax    (name="pool2"  , tensor=net, kernel_size=2)
                
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net, argpool3    = poolargmax    (name="pool3"  , tensor=net, kernel_size=2)
                
                net = tf.cond(self.is_training_pl, lambda: dropout(name="encdrop3", tensor=net, keep_prob=0.5), lambda: net)
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net, argpool4    = poolargmax    (name="pool4"  , tensor=net, kernel_size=2)                
                net = tf.cond(self.is_training_pl, lambda: dropout(name="encdrop4", tensor=net, keep_prob=0.5), lambda: net)                
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net, argpool5    = poolargmax    (name="pool5"  , tensor=net, kernel_size=2)
                net                = tf.cond(self.is_training_pl, lambda: dropout(name="encdrop5", tensor=net, keep_prob=0.5), lambda: net)
    
                # Decoder
                net                = unpool_layer2x2_batch(net, argmax=argpool5)
                # net                = unpoolind        (x=net, shape=[int(h/16), int(w/16)], unpool_mat=argpool5)
                net                = reshape        (tensor=net, shape=[b*s, int(h/16), int(w/16), 512], name="unpool5shape")
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5_D')

                net                = tf.cond(self.is_training, lambda: dropout(name="decdrop5", tensor=net, keep_prob=0.5), lambda: net)
                # net                = unpool_layer2x2_batch(net, argmax=argpool4)
                net                = unpoolind        (x=net, shape=[int(h/8), int(w/8)], unpool_mat=argpool4)
                net                = reshape        (tensor=net, shape=[b*s, int(h/8), int(w/8), 512], name="unpool4shape")
                net                = conv_with_bn2    (name="conv4_3_D", tensor=net, shape=[512, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = conv_with_bn2    (name="conv4_2_D", tensor=net, shape=[512, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = conv_with_bn2    (name="conv4_1_D", tensor=net, shape=[256, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = tf.cond(is_training, lambda: dropout(name="decdrop4", tensor=net, keep_prob=0.5), lambda: net)
                # net                = unpool_layer2x2_batch(net, argmax=argpool3)
                net                = unpoolind        (x=net, shape=[int(h/4), int(w/4)], unpool_mat=argpool3)
                net                = reshape        (tensor=net, shape=[b*s, int(h/4), int(w/4), 256], name="unpool3shape")
                net                = conv_with_bn2    (name="conv3_3_D", tensor=net, shape=[256, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = conv_with_bn2    (name="conv3_2_D", tensor=net, shape=[256, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = conv_with_bn2    (name="conv3_1_D", tensor=net, shape=[128, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = tf.cond(is_training, lambda: dropout(name="decdrop3", tensor=net, keep_prob=0.5), lambda: net)
                # net                = unpool_layer2x2_batch(net, argmax=argpool2)
                net                = unpoolind        (x=net, shape=[int(h/2), int(w/2)], unpool_mat=argpool2)
                net                = reshape        (tensor=net, shape=[b*s, int(h/2), int(w/2), 128], name="unpool2shape")
                net                = conv_with_bn2    (name="conv2_2_D", tensor=net, shape=[128, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = conv_with_bn2    (name="conv2_1_D", tensor=net, shape=[ 64, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                # net                = unpool_layer2x2_batch(net, argmax=argpool1)
                net                = unpoolind        (x=net, shape=[h, w], unpool_mat=argpool1)
                net                = reshape        (tensor=net, shape=[b*s, h, w, 64], name="unpool1shape")
                net                = conv_with_bn2    (name="conv1_2_D", tensor=net, shape=[ 64, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)
                net                = conv_with_bn2    (name="conv1_1_D", tensor=net, shape=[  1, 3, 1], activation="relu", bias=bias, bnfactor=2.0/9)