'''
Created on Oct 8, 2016

@author: george
'''

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf

def weighted_cross_entropy(logits, targets, weights, name=None):

    with tf.name_scope("logistic_loss") as name:
       
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        try:
            targets.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and targets must have the same shape (%s vs %s)"
                           % (logits.get_shape(), targets.get_shape()))

        # The logistic loss formula from above is
        #   x - x * z + log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   -x * z + log(1 + exp(x))
        # Note that these two expressions can be combined into the following:
        #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # To allow computing gradients at zero, we define custom versions of max and
        # abs functions.
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = math_ops.select(cond, logits, zeros)
        neg_abs_logits = math_ops.select(cond, -logits, logits)
        
        return math_ops.mul(weights,math_ops.add(relu_logits - logits * targets,
                            math_ops.log(1 + math_ops.exp(neg_abs_logits))),
                            name=name)