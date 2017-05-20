'''
Created on Dec 1, 2016

@author: george
'''
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from ops.segnet_loss import weighted_per_image_loss
from ops.segnet_loss import weighted_per_image_loss2
from ops.segnet_loss import dist_loss
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
# modules

@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.
EVAL_BATCH_SIZE = 1
BATCH_SIZE = 2
READ_DATA_SIZE = 100
# for CamVid
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 854
IMAGE_DEPTH = 7

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE

TOWER_NAME = "SAMPLETOWER"
def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  print(total_loss)
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

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

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

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

def dump_unravel(indices, shape):
    """
    self-implented unravel indice, missing gradients, need fix
    """
    N = indices.get_shape().as_list()[0]
    tb = tf.constant([shape[0]], shape=[1,N])
    ty = tf.constant([shape[1]], shape=[1,N])
    tx = tf.constant([shape[2]], shape=[1,N])
    tc = tf.constant([shape[3]], shape=[1,N])

    c = indices % tc
    x = ((indices - c) // tc ) % tx
    t_temp = ((indices - c) // tc)
    y = ((t_temp - x) // tx) % ty
    t_temp = ((t_temp - x) // tx)
    b = (t_temp - y) // ty

    t_new = tf.transpose(tf.reshape(tf.pack([b,y,x,c]), (4, N)))
    return t_new

def upsample_with_pool_indices(value, indices, shape=None, scale=2, out_w=None, out_h=None,name="up"):
    s = shape.as_list()
    b = s[0]
    w = s[1]
    h = s[2]
    c = s[3]
    if out_w is not None:
      unraveled = dump_unravel(tf.to_int32(tf.reshape(indices,[b*w*h*c])), [b, out_w, out_h, c])
      ts = tf.SparseTensor(indices=tf.to_int64(unraveled), values=tf.reshape(value, [b*w*h*c]), shape=[b,out_w,out_h,c])
    else:
      unraveled = dump_unravel(tf.to_int32(tf.reshape(indices,[b*w*h*c])), [b, w*scale, h*scale, c])
      ts = tf.SparseTensor(indices=tf.to_int64(unraveled), values=tf.reshape(value, [b*w*h*c]), shape=[b,w*scale,h*scale,c])

    t_dense = tf.sparse_tensor_to_dense(ts, name=name, validate_indices=False)
    return t_dense

def loss(logits, labels):
  """
      loss func without re-weighting
  """
  # Calculate the average cross entropy loss across the batch.
  logits = tf.reshape(logits, (-1,NUM_CLASSES))
  labels = tf.reshape(labels, [-1])

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def weighted_loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):

        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax + epsilon), head), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss

def cal_loss(logits, labels, weights):
    """
    loss_weight = np.asarray([
      0.0616944799702,
      3.89114328416,
      0.718496198987,
      3.24645148591,
      1.64418466389,
      0.0182122198045
    ]) # class 0~5
    """
    """
    loss_weight = np.array([
      0.2595,
      0.1826,
      4.5640,
      0.1417,
      0.9051,
      0.3826,
      9.6446,
      1.8418,
      0.6823,
      6.2478,
      7.3614,
    ]) # class 0~10
    """
 
    
    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    if weights is not None:
        return weighted_per_image_loss2(logits, labels, num_classes = NUM_CLASSES, weight_map=weights)
        #return dist_loss(logits, labels,  dist_map=weights)

    else:
        loss_weight = np.array([
      1.0,
      3.0,
    ]) # class 0~1
        return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

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


def inference(images, labels, phase_train,weights = None):
    #batch_size = BATCH_SIZE
    batch_size = tf.shape(images)[0]
    IMG_HEIGHT = tf.shape(images)[1]#IMAGE_HEIGHT
    IMG_WIDTH = tf.shape(images)[2]#IMAGE_WIDTH
    # norm1
    #norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
    #            name='norm1')
    norm1 = images
    
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, IMAGE_DEPTH, 64], phase_train, name="conv1")
    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, int(IMG_HEIGHT/8), int(IMG_WIDTH/8), 64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, int(IMG_HEIGHT/4), int(IMG_WIDTH/4), 64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, int(IMG_HEIGHT/2), int(IMG_WIDTH/2), 64], 2, "up2")
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, IMG_HEIGHT, IMG_WIDTH, 64], 2, "up1")
    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, NUM_CLASSES],
                                           initializer=msra_initializer(1, 64),
                                           wd=0.00001)
        conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if labels is None:
        return logit
    else:
        loss = cal_loss(conv_classifier, labels,weights)
        return loss, logit


def inference_vgg16(images, labels, phase_train, weights=None):
    # batch_size = BATCH_SIZE
    batch_size = tf.shape(images)[0]
    IMG_HEIGHT = tf.shape(images)[1]
    IMG_WIDTH = tf.shape(images)[2] # IMAGE_WIDTH
    # norm1
    # norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
    #            name='norm1')
    norm1 = images

    # conv1
    conv1_1 = conv_layer_with_bn(norm1, [3, 3, IMAGE_DEPTH, 64], phase_train, name="conv1_1")
    conv1_2 = conv_layer_with_bn(conv1_1, [3, 3, 64, 64], phase_train, name="conv1_2")

    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool1')

    # conv2
    conv2_1 = conv_layer_with_bn(pool1, [3, 3, 64, 128], phase_train, name="conv2_1")
    conv2_2 = conv_layer_with_bn(conv2_1, [3, 3, 128,128], phase_train, name="conv2_2")

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3_1 = conv_layer_with_bn(pool2, [3, 3, 128, 256], phase_train, name="conv3_1")
    conv3_2 = conv_layer_with_bn(conv3_1, [3, 3, 256, 256], phase_train, name="conv3_2")
    conv3_3 = conv_layer_with_bn(conv3_2, [3, 3, 256, 256], phase_train, name="conv3_3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4_1 = conv_layer_with_bn(pool3, [3, 3, 256, 512], phase_train, name="conv4_1")
    conv4_2 = conv_layer_with_bn(conv4_1, [3, 3, 512, 512], phase_train, name="conv4_2")
    conv4_3 = conv_layer_with_bn(conv4_2, [3, 3, 512, 512], phase_train, name="conv4_3")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    conv5_1 = conv_layer_with_bn(pool4, [3, 3, 512, 512], phase_train, name="conv5_1")
    conv5_2 = conv_layer_with_bn(conv5_1, [3, 3, 512, 512], phase_train, name="conv5_2")
    conv5_3 = conv_layer_with_bn(conv5_2, [3, 3, 512, 512], phase_train, name="conv5_3")


    # pool5
    pool5, pool5_indices = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    print(pool5)
    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    upsample5 = deconv_layer(pool5, [2, 2, 512, 512], [batch_size, int(np.ceil(IMG_HEIGHT / 16)), int(np.ceil(IMG_WIDTH / 16)), 512], 2, "up5")
    conv_decode5_3 = conv_layer_with_bn(upsample5, [3, 3, 512, 512], phase_train, False, name="conv_decode5_3")
    conv_decode5_2 = conv_layer_with_bn(conv_decode5_3, [3, 3, 512, 512], phase_train, False, name="conv_decode5_2")
    conv_decode5_1 = conv_layer_with_bn(conv_decode5_2, [3, 3, 512, 512], phase_train, False, name="conv_decode5_1")

    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(conv_decode5_1, [2, 2, 512, 512], [batch_size, int(IMG_HEIGHT / 8), int(IMG_WIDTH / 8), 512], 2, "up4")
    # decode 4
    conv_decode4_3 = conv_layer_with_bn(upsample4, [3, 3, 512, 512], phase_train, False, name="conv_decode4_3")
    conv_decode4_2 = conv_layer_with_bn(conv_decode4_3, [3, 3, 512, 512], phase_train, False, name="conv_decode4_2")
    conv_decode4_1 = conv_layer_with_bn(conv_decode4_2, [3, 3, 512, 256], phase_train, False, name="conv_decode4_1")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3 = deconv_layer(conv_decode4_1, [2, 2, 256, 256], [batch_size, int(IMG_HEIGHT / 4), int(IMG_WIDTH / 4), 256], 2,
                             "up3")
    # decode 3
    conv_decode3_3 = conv_layer_with_bn(upsample3, [3, 3, 256, 256], phase_train, False, name="conv_decode3_3")
    conv_decode3_2 = conv_layer_with_bn(conv_decode3_3, [3, 3, 256, 256], phase_train, False, name="conv_decode3_2")
    conv_decode3_1 = conv_layer_with_bn(conv_decode3_2, [3, 3, 256, 128], phase_train, False, name="conv_decode3_1")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2 = deconv_layer(conv_decode3_1, [2, 2, 128, 128], [batch_size, int(IMG_HEIGHT / 2), int(IMG_WIDTH / 2), 128], 2,
                             "up2")
    # decode 2
    conv_decode2_2 = conv_layer_with_bn(upsample2, [3, 3, 128, 128], phase_train, False, name="conv_decode2_2")
    conv_decode2_1 = conv_layer_with_bn(conv_decode2_2, [3, 3, 128, 64], phase_train, False, name="conv_decode2_1")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1 = deconv_layer(conv_decode2_1, [2, 2, 64, 64], [batch_size, IMG_HEIGHT, IMG_WIDTH, 64], 2, "up1")
    # decode4
    conv_decode1_2 = conv_layer_with_bn(upsample1, [3, 3, 64, 64], phase_train, False, name="conv_decode1_2")
    conv_decode1_1 = conv_layer_with_bn(conv_decode1_2, [3, 3, 64, 64], phase_train, False, name="conv_decode1_1")

    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.00001)
        conv = tf.nn.conv2d(conv_decode1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if labels is None:
        return logit
    else:
        loss = cal_loss(conv_classifier, labels, weights)
        return loss, logit

def inference_encoder_decoder(images,phase_train):
    # batch_size = BATCH_SIZE
    batch_size = tf.shape(images)[0]
    IMG_HEIGHT = tf.shape(images)[1]#IMAGE_HEIGHT
    IMG_WIDTH = tf.shape(images)[2]#IMAGE_WIDTH
    # norm1
    # norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
    #            name='norm1')
    norm1 = images

    # conv1
    conv1_1 = conv_layer_with_bn(norm1, [3, 3, IMAGE_DEPTH, 64], phase_train, name="conv1_1")
    conv1_2 = conv_layer_with_bn(conv1_1, [3, 3, 64, 64], phase_train, name="conv1_2")

    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool1')

    # conv2
    conv2_1 = conv_layer_with_bn(pool1, [3, 3, 64, 128], phase_train, name="conv2_1")
    conv2_2 = conv_layer_with_bn(conv2_1, [3, 3, 128, 128], phase_train, name="conv2_2")

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3_1 = conv_layer_with_bn(pool2, [3, 3, 128, 256], phase_train, name="conv3_1")
    conv3_2 = conv_layer_with_bn(conv3_1, [3, 3, 256, 256], phase_train, name="conv3_2")
    conv3_3 = conv_layer_with_bn(conv3_2, [3, 3, 256, 256], phase_train, name="conv3_3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4_1 = conv_layer_with_bn(pool3, [3, 3, 256, 512], phase_train, name="conv4_1")
    conv4_2 = conv_layer_with_bn(conv4_1, [3, 3, 512, 512], phase_train, name="conv4_2")
    conv4_3 = conv_layer_with_bn(conv4_2, [3, 3, 512, 512], phase_train, name="conv4_3")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    conv5_1 = conv_layer_with_bn(pool4, [3, 3, 512, 512], phase_train, name="conv5_1")
    conv5_2 = conv_layer_with_bn(conv5_1, [3, 3, 512, 512], phase_train, name="conv5_2")
    conv5_3 = conv_layer_with_bn(conv5_2, [3, 3, 512, 512], phase_train, name="conv5_3")

    # pool5
    pool5, pool5_indices = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    out_size_5 = tf.stack([batch_size,tf.cast(tf.ceil(IMG_HEIGHT / 16),tf.int32),tf.cast(tf.ceil(IMG_WIDTH / 16),tf.int32),512])

    upsample5 = deconv_layer(pool5, [2, 2, 512, 512],out_size_5, 2, "up5")
    concat_5 = tf.concat( [upsample5, conv5_3],3, name='concat5')

    conv_decode5_3 = conv_layer_with_bn(concat_5, [3, 3, 1024, 512], phase_train, False, name="conv_decode5_3")
    conv_decode5_2 = conv_layer_with_bn(conv_decode5_3, [3, 3, 512, 512], phase_train, False, name="conv_decode5_2")
    conv_decode5_1 = conv_layer_with_bn(conv_decode5_2, [3, 3, 512, 512], phase_train, False, name="conv_decode5_1")

    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    out_size_4 = tf.stack([batch_size,tf.cast(tf.ceil(IMG_HEIGHT / 8),tf.int32),tf.cast(tf.ceil(IMG_WIDTH / 8),tf.int32),512])

    upsample4 = deconv_layer(conv_decode5_1, [2, 2, 512, 512],out_size_4, 2, "up4")
    concat_4 = tf.concat([upsample4, conv4_3],3, name='concat4')

    # decode 4
    conv_decode4_3 = conv_layer_with_bn(concat_4, [3, 3, 1024, 512], phase_train, False, name="conv_decode4_3")
    conv_decode4_2 = conv_layer_with_bn(conv_decode4_3, [3, 3, 512, 512], phase_train, False, name="conv_decode4_2")
    conv_decode4_1 = conv_layer_with_bn(conv_decode4_2, [3, 3, 512, 256], phase_train, False, name="conv_decode4_1")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    out_size_3 = tf.stack([batch_size,tf.cast(tf.ceil(IMG_HEIGHT / 4),tf.int32),tf.cast(tf.ceil(IMG_WIDTH / 4),tf.int32),256])

    upsample3 = deconv_layer(conv_decode4_1, [2, 2, 256, 256],out_size_3, 2,
                             "up3")
    concat_3 = tf.concat([upsample3, conv3_3],3, name='concat3')

    # decode 3
    conv_decode3_3 = conv_layer_with_bn(concat_3, [3, 3, 512, 256], phase_train, False, name="conv_decode3_3")
    conv_decode3_2 = conv_layer_with_bn(conv_decode3_3, [3, 3, 256, 256], phase_train, False, name="conv_decode3_2")
    conv_decode3_1 = conv_layer_with_bn(conv_decode3_2, [3, 3, 256, 128], phase_train, False, name="conv_decode3_1")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    out_size_2 = tf.stack([batch_size,tf.cast(tf.ceil(IMG_HEIGHT / 2),tf.int32),tf.cast(tf.ceil(IMG_WIDTH / 2),tf.int32),128])

    upsample2 = deconv_layer(conv_decode3_1, [2, 2, 128, 128],out_size_2, 2,
                             "up2")
    concat_2 = tf.concat([upsample2, conv2_2],3, name='concat2')

    # decode 2
    conv_decode2_2 = conv_layer_with_bn(concat_2, [3, 3, 256, 128], phase_train, False, name="conv_decode2_2")
    conv_decode2_1 = conv_layer_with_bn(conv_decode2_2, [3, 3, 128, 64], phase_train, False, name="conv_decode2_1")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    out_size_1 = tf.stack([batch_size,IMG_HEIGHT,IMG_WIDTH ,64])

    upsample1 = deconv_layer(conv_decode2_1, [2, 2, 64, 64], out_size_1, 2, "up1")
    concat_1 = tf.concat([upsample1, conv1_2],3, name='concat1')

    # decode4
    conv_decode1_2 = conv_layer_with_bn(concat_1, [3, 3, 128, 64], phase_train, False, name="conv_decode1_2")
    conv_decode1_1 = conv_layer_with_bn(conv_decode1_2, [3, 3, 64, 64], phase_train, False, name="conv_decode1_1")
    return conv_decode1_1

def inference_vgg16_withskip(images, labels, phase_train, weights=None):

    conv_decode1_1 = inference_encoder_decoder(images,phase_train)
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.0001)
        conv = tf.nn.conv2d(conv_decode1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if labels is None:
        return logit
    else:
        loss = cal_loss(conv_classifier, labels, weights)
        return loss, logit


def inference_merge_two_branch(images_branch1,images_branch2, labels, phase_train, weights=None):
    with tf.variable_scope('branch1') as scope:
        branch1 = inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        branch2 = inference_encoder_decoder(images_branch2,phase_train)

    with tf.variable_scope('merger') as scope:
        net = tf.concat(3, [branch1, branch2], name='merged_b1_b2')
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = conv_layer_with_bn(net, [3, 3, 128, 64], phase_train, False, name="merge_conv1")
        net = conv_layer_with_bn(net, [3, 3, 64, 64], phase_train, False, name="merge_conv2")
        net = conv_layer_with_bn(net, [3, 3, 64, 32], phase_train, False, name="merge_conv3")


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 32, NUM_CLASSES],
                                             initializer=msra_initializer(1, 32),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if labels is None:
        return logit
    else:
        loss = cal_loss(conv_classifier, labels, weights)
        return loss, logit

def initialize_merge_net(sess,checkpoint_branch1,checkpoint_branch2):
    all_vars = tf.all_variables()
    print([v.op.name for v in all_vars])
    branch1_vars = {v.op.name.replace("branch1/",""):v for v in all_vars if v.name.startswith("branch1")}
    branch2_vars = {v.op.name.replace("branch2/",""):v for v in all_vars if v.name.startswith("branch2")}
    print(branch1_vars)
    print(branch2_vars)

    saver_branch1 = tf.train.Saver(branch1_vars)
    saver_branch2 = tf.train.Saver(branch2_vars)

    saver_branch1.restore(sess, checkpoint_branch1)
    saver_branch2.restore(sess, checkpoint_branch2)

def inference_vgg16_withdrop(images, labels, phase_train, weights=None,keep_prob = 1.0):
    # batch_size = BATCH_SIZE
    batch_size = tf.shape(images)[0]
    IMG_HEIGHT = IMAGE_HEIGHT
    IMG_WIDTH = IMAGE_WIDTH
    # norm1
    # norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
    #            name='norm1')
    norm1 = images

    # conv1
    conv1_1 = conv_layer_with_bn(norm1, [3, 3, IMAGE_DEPTH, 64], phase_train, name="conv1_1")
    conv1_2 = conv_layer_with_bn(conv1_1, [3, 3, 64, 64], phase_train, name="conv1_2")

    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='pool1')

    # conv2
    conv2_1 = conv_layer_with_bn(pool1, [3, 3, 64, 128], phase_train, name="conv2_1")
    conv2_2 = conv_layer_with_bn(conv2_1, [3, 3, 128, 128], phase_train, name="conv2_2")

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3_1 = conv_layer_with_bn(pool2, [3, 3, 128, 256], phase_train, name="conv3_1")
    conv3_2 = conv_layer_with_bn(conv3_1, [3, 3, 256, 256], phase_train, name="conv3_2")
    conv3_3 = conv_layer_with_bn(conv3_2, [3, 3, 256, 256], phase_train, name="conv3_3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    pool3 = tf.nn.dropout(pool3,keep_prob)

    # conv4
    conv4_1 = conv_layer_with_bn(pool3, [3, 3, 256, 512], phase_train, name="conv4_1")
    conv4_2 = conv_layer_with_bn(conv4_1, [3, 3, 512, 512], phase_train, name="conv4_2")
    conv4_3 = conv_layer_with_bn(conv4_2, [3, 3, 512, 512], phase_train, name="conv4_3")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    pool4 = tf.nn.dropout(pool4,keep_prob)

    conv5_1 = conv_layer_with_bn(pool4, [3, 3, 512, 512], phase_train, name="conv5_1")
    conv5_2 = conv_layer_with_bn(conv5_1, [3, 3, 512, 512], phase_train, name="conv5_2")
    conv5_3 = conv_layer_with_bn(conv5_2, [3, 3, 512, 512], phase_train, name="conv5_3")

    # pool5
    pool5, pool5_indices = tf.nn.max_pool_with_argmax(conv5_3, ksize=[1, 2, 2, 1],
                                                      strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    pool5 = tf.nn.dropout(pool5,keep_prob)

    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    upsample5 = deconv_layer(pool5, [2, 2, 512, 512],
                             [batch_size, int(ceil(IMG_HEIGHT / 16)), int(np.ceil(IMG_WIDTH / 16)), 512], 2, "up5")
    concat_5 = tf.concat(3, [upsample5, conv5_3], name='concat5')

    conv_decode5_3 = conv_layer_with_bn(concat_5, [3, 3, 1024, 512], phase_train, False, name="conv_decode5_3")
    conv_decode5_2 = conv_layer_with_bn(conv_decode5_3, [3, 3, 512, 512], phase_train, False, name="conv_decode5_2")
    conv_decode5_1 = conv_layer_with_bn(conv_decode5_2, [3, 3, 512, 512], phase_train, False, name="conv_decode5_1")

    conv_decode5_1 =  tf.nn.dropout(conv_decode5_1,keep_prob)
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')

    upsample4 = deconv_layer(conv_decode5_1, [2, 2, 512, 512],
                             [batch_size, int(IMG_HEIGHT / 8), int(IMG_WIDTH / 8), 512], 2, "up4")
    concat_4 = tf.concat(3, [upsample4, conv4_3], name='concat4')

    # decode 4
    conv_decode4_3 = conv_layer_with_bn(concat_4, [3, 3, 1024, 512], phase_train, False, name="conv_decode4_3")
    conv_decode4_2 = conv_layer_with_bn(conv_decode4_3, [3, 3, 512, 512], phase_train, False, name="conv_decode4_2")
    conv_decode4_1 = conv_layer_with_bn(conv_decode4_2, [3, 3, 512, 256], phase_train, False, name="conv_decode4_1")

    conv_decode4_1 =  tf.nn.dropout(conv_decode4_1,keep_prob)

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3 = deconv_layer(conv_decode4_1, [2, 2, 256, 256],
                             [batch_size, int(IMG_HEIGHT / 4), int(IMG_WIDTH / 4), 256], 2,
                             "up3")
    concat_3 = tf.concat(3, [upsample3, conv3_3], name='concat3')

    # decode 3
    conv_decode3_3 = conv_layer_with_bn(concat_3, [3, 3, 512, 256], phase_train, False, name="conv_decode3_3")
    conv_decode3_2 = conv_layer_with_bn(conv_decode3_3, [3, 3, 256, 256], phase_train, False, name="conv_decode3_2")
    conv_decode3_1 = conv_layer_with_bn(conv_decode3_2, [3, 3, 256, 128], phase_train, False, name="conv_decode3_1")

    conv_decode3_1 =  tf.nn.dropout(conv_decode3_1,keep_prob)

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2 = deconv_layer(conv_decode3_1, [2, 2, 128, 128],
                             [batch_size, int(IMG_HEIGHT / 2), int(IMG_WIDTH / 2), 128], 2,
                             "up2")
    concat_2 = tf.concat(3, [upsample2, conv2_2], name='concat2')

    # decode 2
    conv_decode2_2 = conv_layer_with_bn(concat_2, [3, 3, 256, 128], phase_train, False, name="conv_decode2_2")
    conv_decode2_1 = conv_layer_with_bn(conv_decode2_2, [3, 3, 128, 64], phase_train, False, name="conv_decode2_1")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1 = deconv_layer(conv_decode2_1, [2, 2, 64, 64], [batch_size, IMG_HEIGHT, IMG_WIDTH, 64], 2, "up1")
    concat_1 = tf.concat(3, [upsample1, conv1_2], name='concat1')

    # decode4
    conv_decode1_2 = conv_layer_with_bn(concat_1, [3, 3, 128, 64], phase_train, False, name="conv_decode1_2")
    conv_decode1_1 = conv_layer_with_bn(conv_decode1_2, [3, 3, 64, 64], phase_train, False, name="conv_decode1_1")

    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=msra_initializer(1, 64),
                                             wd=0.00001)
        conv = tf.nn.conv2d(conv_decode1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if labels is None:
        return logit
    else:
        loss = cal_loss(conv_classifier, labels, weights)
        return loss, logit

def prepare_encoder_parameters():
    param_format = 'conv%d_%d'
    conv_layers = [2, 2, 3, 3, 3]
    params = []

    for pool in range(1, 6):
        for conv in range(1, conv_layers[pool - 1] + 1):

            with tf.variable_scope(param_format % (pool,conv), reuse=True):
                  weights = tf.get_variable('ort_weights')
                  biases = tf.get_variable('biases')
                  params += [weights, biases]

    return params

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

def initialize_vgg16(sess):
    params = prepare_encoder_parameters()
    WEIGHT_FILE = 'checkpoints/vgg16_weights.npz'
    weights = np.load(WEIGHT_FILE)
    keys = sorted(weights.keys())[:26]
    for i, k in enumerate(keys):
        val = weights[k]
        if k=='conv1_1_W':
            print('applying modified %s',k)
            # Add extra channel for previous mask channel
            prev_mask_filt_array = get_init_prev_mask_filters(val)
            temp_mod_val = np.concatenate((val, prev_mask_filt_array), axis=2)
            # Add extra channel for previous mask channel
            prev_rgb_array = get_init_prev_rgb_filters(val)
            temp_mod_val = np.concatenate((temp_mod_val, prev_rgb_array), axis=2)
            sess.run(params[i].assign(temp_mod_val))

        else:
            print('applying  %s',k)
            sess.run(params[i].assign(val))


def train(total_loss, global_step):
    batch_size = BATCH_SIZE
    total_sample = 274
    num_batches_per_epoch = 274/1
    """ fix lr """
    lr = INITIAL_LEARNING_RATE
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_hist(predictions, labels):
  hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
  for i in range(BATCH_SIZE):
    hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), NUM_CLASSES)
  return hist

def print_hist_summery(hist):
  acc_total = np.diag(hist).sum() / hist.sum()
  print ('accuracy = %f'%np.nanmean(acc_total))
  iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
  print ('mean IU  = %f'%np.nanmean(iu))
  for ii in range(NUM_CLASSES):
      if float(hist.sum(1)[ii]) == 0:
        acc = 0.0
      else:
        acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
      print("    class # %d accuracy = %f "%(ii, acc))

def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    num_class = NUM_CLASSES
    size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f'%np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f'%np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0
        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f "%(ii,acc))

def eval_batches(data, sess, eval_prediction=None):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0] # batch_size
    predictions = np.ndarray(shape=(size, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES), dtype=np.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = eval_prediction
      else:
        batch_predictions = eval_prediction
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions
