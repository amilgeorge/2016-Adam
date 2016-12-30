'''
Created on Dec 18, 2016

@author: george
'''

import tensorflow as tf
from tensorflow.python.framework import ops

def weighted_per_image_loss(logits, labels, num_classes, weight_map=None):
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
        
        # Weights flat
        weights_flat = tf.reshape(weight_map,(-1,1))
        
        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        #cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax + epsilon), [1.0,1.0]), reduction_indices=[1])
        tmp1 = tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1])
        reshaped_tmp = tf.reshape(tmp1,(-1,1))

        cross_entropy = -tf.mul(reshaped_tmp,weights_flat)

        #cross_entropy = -tf.mul(tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1]),weights_flat)

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss

    def softmax_accuracy():
        pass