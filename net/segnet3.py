'''
Created on Dec 1, 2016

@author: george
'''
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
#from ops.segnet_loss import weighted_per_image_loss
#from ops.segnet_loss import weighted_per_image_loss2
#from ops.segnet_loss import dist_loss
import os, sys
import numpy as np
import math
from datetime import datetime
import time
from math import ceil
from tensorflow.python.ops import gen_nn_ops
import skimage
import skimage.io
import re
from models import resnet_v1
slim = tf.contrib.slim

# modules

IMG_HEIGHT = 360
IMG_WIDTH = 480
NUM_CLASSES = 2
BATCH_SIZE = 4


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
      """
      kernel = _variable_with_weight_decay('weights',
                                           shape=shape,
                                           initializer=msra_initializer(k_size, in_channel),
                                           wd=None)
      """
      kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
      Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
      Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, initializer, wd):
    """Helper to create an initialized Variable with weight decay.
      Note that the Variable is initialized with a truncated normal distribution.
      A weight decay is added only if one is specified.
      Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
      Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(name,
                           shape,
                           initializer)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def orthogonal_initializer(scale = 1.1,partition_info = None):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32,partition_info = None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def inference2(images, phase_train, is_training = True,weights=None):
    batch_size = tf.shape(images)[0]
    weight_decay = 0.0005

    # encoder
    with slim.arg_scope(resnet_v1.resnet_arg_scope2(is_training=phase_train)):
        net, end_points_head = resnet_v1.resnet_v1_50(images,
                                                      global_pool=False,
                                                    output_stride=16)

    # Decoder

    ### Skip layers
    skip_8 = end_points_head['resnet_v1_50/block1'] #256
    skip_2 = end_points_head['resnet_v1_50/conv1']
    ###

    net = conv_layer_with_bn(net, [1, 1, 2048, 1024], phase_train, False, name="decode_reduce_4_2")
    net = conv_layer_with_bn(net, [1, 1, 1024, 512], phase_train, False, name="decode_reduce_4_1")


    net = deconv_layer(net, [2, 2, 512, 512],
                                 [batch_size, int(ceil(IMG_HEIGHT / 8)), int(np.ceil(IMG_WIDTH / 8)), 512], 2, "up2x")

    net = conv_layer_with_bn(net, [1, 1, 512, 256], phase_train, False, name="decode_reduce_3_2")
    net = tf.concat(3, [net,skip_8], name='concat3')
    net = conv_layer_with_bn(net, [1, 1, 512, 256], phase_train, False, name="decode_reduce_3_1")

    net = deconv_layer(net, [2, 2, 256, 256],
                       [batch_size, int(ceil(IMG_HEIGHT / 4)), int(np.ceil(IMG_WIDTH / 4)), 256], 2, "up4x")

    net = conv_layer_with_bn(net, [1, 1, 256, 256], phase_train, False, name="decode_reduce_2_2")
    net = conv_layer_with_bn(net, [1, 1, 256, 128], phase_train, False, name="decode_reduce_2_1")


    net = deconv_layer(net, [2, 2, 128, 128],
                       [batch_size, int(ceil(IMG_HEIGHT / 2)), int(np.ceil(IMG_WIDTH / 2)), 128], 2, "up8x")

    net = conv_layer_with_bn(net, [1, 1, 128, 64], phase_train, False, name="decode_reduce_1_2")
    net = tf.concat(3, [net, skip_2], name='concat3')
    net = conv_layer_with_bn(net, [1, 1, 128, 64], phase_train, False, name="decode_reduce_1_1")

    net = deconv_layer(net, [2, 2, 64, 64],
                       [batch_size, int(IMG_HEIGHT), int(IMG_WIDTH), 64], 2, "up16x")

    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.0001)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier

    return  logit

def inference(images, phase_train, is_training = True,weights=None):
    batch_size = tf.shape(images)[0]
    weight_decay = 0.0005

    # encoder
    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=is_training, weight_decay=weight_decay)):
        net, end_points_head = resnet_v1.resnet_v1_50(images,
                                                      global_pool=False,
                                                    output_stride=16)

    # Decoder

    ### Skip layers
    skip_8 = end_points_head['resnet_v1_50/block1'] #256
    skip_2 = end_points_head['resnet_v1_50/conv1']
    ###

    net = conv_layer_with_bn(net, [1, 1, 2048, 1024], phase_train, False, name="decode_reduce_4_2")
    net = conv_layer_with_bn(net, [1, 1, 1024, 512], phase_train, False, name="decode_reduce_4_1")


    net = deconv_layer(net, [2, 2, 512, 512],
                                 [batch_size, int(ceil(IMG_HEIGHT / 8)), int(np.ceil(IMG_WIDTH / 8)), 512], 2, "up2x")

    net = conv_layer_with_bn(net, [1, 1, 512, 256], phase_train, False, name="decode_reduce_3_2")
    net = tf.concat(3, [net,skip_8], name='concat3')
    net = conv_layer_with_bn(net, [1, 1, 512, 256], phase_train, False, name="decode_reduce_3_1")

    net = deconv_layer(net, [2, 2, 256, 256],
                       [batch_size, int(ceil(IMG_HEIGHT / 4)), int(np.ceil(IMG_WIDTH / 4)), 256], 2, "up4x")

    net = conv_layer_with_bn(net, [1, 1, 256, 256], phase_train, False, name="decode_reduce_2_2")
    net = conv_layer_with_bn(net, [1, 1, 256, 128], phase_train, False, name="decode_reduce_2_1")


    net = deconv_layer(net, [2, 2, 128, 128],
                       [batch_size, int(ceil(IMG_HEIGHT / 2)), int(np.ceil(IMG_WIDTH / 2)), 128], 2, "up8x")

    net = conv_layer_with_bn(net, [1, 1, 128, 64], phase_train, False, name="decode_reduce_1_2")
    net = tf.concat(3, [net, skip_2], name='concat3')
    net = conv_layer_with_bn(net, [1, 1, 128, 64], phase_train, False, name="decode_reduce_1_1")

    net = deconv_layer(net, [2, 2, 64, 64],
                       [batch_size, int(IMG_HEIGHT), int(IMG_WIDTH), 64], 2, "up16x")

    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.0001)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier

    return  logit

def get_init_prev_mask_filters(temp_val):
    num_filters = temp_val.shape[-1]
    kernal_size = temp_val.shape[0:2]
    filt_list = []

    for i in range(num_filters):
        mean_filt = temp_val[:, :, :, i].mean()
        std_filt = temp_val[:, :, :, i].std()

        filt = np.random.normal(mean_filt, std_filt, kernal_size)
        filt = filt[:, :, np.newaxis, np.newaxis]
        filt_list.append(filt)

    filt_array = np.concatenate(filt_list, axis=3)
    return filt_array


def get_init_prev_rgb_filters(temp_val):
    is_same = True
    if (is_same):
        return temp_val

    num_filters = temp_val.shape[-1]
    kernal_size = temp_val.shape[0:2]
    filt_list = []

    for i in range(num_filters):
        mean_rfilt = temp_val[:, :, 0, i].mean()
        std_rfilt = temp_val[:, :, 0, i].std()

        mean_gfilt = temp_val[:, :, 1, i].mean()
        std_gfilt = temp_val[:, :, 1, i].std()

        mean_bfilt = temp_val[:, :, 2, i].mean()
        std_bfilt = temp_val[:, :, 2, i].std()

        rfilt = np.random.normal(mean_rfilt, std_rfilt, kernal_size)
        gfilt = np.random.normal(mean_gfilt, std_gfilt, kernal_size)
        bfilt = np.random.normal(mean_bfilt, std_bfilt, kernal_size)
        rgb_filt = np.dstack((rfilt, gfilt, bfilt))
        rgb_filt = rgb_filt[:, :, :, np.newaxis]
        filt_list.append(rgb_filt)

    filt_array = np.concatenate(filt_list, axis=3)
    return filt_array


def initialize_resnet(sess):
    WEIGHT_FILE = 'checkpoints/resnet_v1_50.ckpt'
    reader = tf.train.NewCheckpointReader(WEIGHT_FILE)

    var_to_shape_map = reader.get_variable_to_shape_map()
    with tf.variable_scope('resnet_v1_50', reuse=True):

        for key in var_to_shape_map:

            if not key.startswith('resnet_v1_50'):
                print('skipping {}'.format(key))
                continue

            sub_key = key[13:]
            try:
                t = tf.get_variable(sub_key)
                val = reader.get_tensor(key)
                if key =='resnet_v1_50/conv1/weights':
                    print('applying modified %s', key)
                    prev_mask_filt_array = get_init_prev_mask_filters(val)
                    temp_mod_val = np.concatenate((val, prev_mask_filt_array), axis=2)
                    # Add extra channel for previous mask channel
                    prev_rgb_array = get_init_prev_rgb_filters(val)
                    temp_mod_val = np.concatenate((temp_mod_val, prev_rgb_array), axis=2)
                    sess.run(t.assign(temp_mod_val))
                else:
                    print('applying  %s', key)
                    sess.run(t.assign(val))

            except:
                print ("Exception for {}".format(key))


"""
    keys = sorted(weights.keys())[:26]


    for i, k in enumerate(keys):
        val = weights[k]
        if k == 'conv1_1_W':
            print('applying modified %s', k)
            # Add extra channel for previous mask channel
            prev_mask_filt_array = get_init_prev_mask_filters(val)
            temp_mod_val = np.concatenate((val, prev_mask_filt_array), axis=2)
            # Add extra channel for previous mask channel
            prev_rgb_array = get_init_prev_rgb_filters(val)
            temp_mod_val = np.concatenate((temp_mod_val, prev_rgb_array), axis=2)
            sess.run(params[i].assign(temp_mod_val))

        else:
            print('applying  %s', k)
            sess.run(params[i].assign(val))
"""
if __name__ == '__main__':
    inp = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 7], name='input')
    label = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH], name='label')
    weights = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH], name='weights')
    keep_prob = tf.placeholder(tf.float32)
    is_training_pl = tf.placeholder(tf.bool, name="segnet_is_training")

    net,end_points = inference(inp,is_training_pl)