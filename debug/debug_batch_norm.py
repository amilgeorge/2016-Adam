'''
Created on Oct 12, 2016

@author: george
'''

import collections
import tensorflow as tf

slim = tf.contrib.slim
import models.resnet_utils as resnetutils
from dataprovider.sampleinputprovider import SampleInputProvider
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

def batch_norm3(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"
    Sergey Ioffe, Christian Szegedy
  Can be used as a normalizer function for conv2d and fully_connected.
  Args:
    inputs: a tensor of size `[batch_size, height, width, channels]`
            or `[batch_size, channels]`.
    decay: decay for the moving average.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Optional activation function.
    updates_collections: collections to collect the update ops for computation.
      If None, a control dependency would be added to make sure the updates are
      computed.
    is_training: whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_op_scope`.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """
  with variable_scope.variable_op_scope([inputs],
                                        scope, 'BatchNorm', reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = list(range(inputs_rank - 1))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=init_ops.ones_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections.
    moving_mean_collections = utils.get_variable_collections(
        variables_collections, 'moving_mean')
    moving_mean = variables.model_variable(
        'moving_mean',
        shape=params_shape,
        dtype=dtype,
        initializer=init_ops.zeros_initializer,
        trainable=False,
        collections=moving_mean_collections)
    moving_variance_collections = utils.get_variable_collections(
        variables_collections, 'moving_variance')
    moving_variance = variables.model_variable(
        'moving_variance',
        shape=params_shape,
        dtype=dtype,
        initializer=init_ops.ones_initializer,
        trainable=False,
        collections=moving_variance_collections)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
    # Calculate the moments based on the individual batch.
      mean, variance = nn.moments(inputs, axis, shift=moving_mean)
      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            return array_ops.identity(mean), array_ops.identity(variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        vars_fn = lambda: (mean, variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    # Compute batch_normalization.
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


  
def batch_norm2(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"
    Sergey Ioffe, Christian Szegedy
  Can be used as a normalizer function for conv2d and fully_connected.
  Args:
    inputs: a tensor of size `[batch_size, height, width, channels]`
            or `[batch_size, channels]`.
    decay: decay for the moving average.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Optional activation function.
    updates_collections: collections to collect the update ops for computation.
      If None, a control dependency would be added to make sure the updates are
      computed.
    is_training: whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_op_scope`.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """
  with variable_scope.variable_op_scope([inputs],
                                        scope, 'BatchNorm', reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = list(range(inputs_rank - 1))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                      'beta')
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=init_ops.ones_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections.
    moving_mean_collections = utils.get_variable_collections(
        variables_collections, 'moving_mean')
    moving_mean = variables.model_variable(
        'moving_mean',
        shape=params_shape,
        dtype=dtype,
        initializer=init_ops.zeros_initializer,
        trainable=False,
        collections=moving_mean_collections)
    moving_variance_collections = utils.get_variable_collections(
        variables_collections, 'moving_variance')
    moving_variance = variables.model_variable(
        'moving_variance',
        shape=params_shape,
        dtype=dtype,
        initializer=init_ops.ones_initializer,
        trainable=False,
        collections=moving_variance_collections)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
    # Calculate the moments based on the individual batch.
      mean, variance = nn.moments(inputs, axis, shift=moving_mean)
      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            return array_ops.identity(mean), array_ops.identity(variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        vars_fn = lambda: (mean, variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    # Compute batch_normalization.
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

def resnet_arg_scope(weight_decay=0.0000,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections':  tf.GraphKeys.UPDATE_OPS,
  }
      
      
  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=batch_norm3,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc
    
def build_model(inp,is_training):
    with slim.arg_scope(resnet_arg_scope()):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):   
            net = resnetutils.conv2d_same(inp, 64, 7, stride=2, scope='conv1') 
    
    return net
if __name__ == '__main__':
    inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
    ### slim.batch_norm
    # True - 0.3376
    # False - 50.92886
    
    ### batch_norm2
    ## True - 0.3376
    ## False - 0.3376
    
    net = build_model(inp, False)
    mean = tf.reduce_mean(net)
    inputProvider = SampleInputProvider(is_coarse=True,is_dummy=True)    
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver(max_to_keep = 10)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #sess.run(init_op)
        saver.restore(sess, 'bnorm.ckpt')
        while True:
            next_batch = inputProvider.sequence_batch_itr(16)
            for i, sequence_batch in enumerate(next_batch):
                result = sess.run([mean], 
                                      feed_dict={inp:sequence_batch.images})
                
                #saver.save(sess, 'bnorm.ckpt')
                print("Done")