'''
Created on Dec 18, 2016

@author: george
'''

import tensorflow as tf
from tensorflow.python.framework import ops


def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def get_indicator_tensor(indices,shape):
    print(indices)
    #indicator_values = tf.tile(tf.constant(1), indices.get_shape()[0])
    indicator_values = tf.ones(tf.shape(indices)[0])

    sparse_tensor = tf.SparseTensor(indices=indices, values=indicator_values, shape=shape)
    dense_tensor = tf.sparse_tensor_dense_matmul(sparse_tensor,tf.ones(shape))
    #dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor)
    #dense_tensor = tf.sparse_to_dense(indices,shape,indicator_values)

    return dense_tensor

def get_mulmat(logits_flat, labels_flat, dist_map_flat):

    total_fg_count = tf_count(labels_flat, 1)
    total_bg_count = tf_count(labels_flat, 0)


    bg_count = tf.minimum(total_bg_count,3*total_fg_count)

    out = tf.nn.softmax(logits_flat)

    prob_map = out[:,1:2]

    # select hard negatives
    neg_locs = 1 - labels_flat
    only_neg_maps = tf.multiply(prob_map, tf.cast(neg_locs,tf.float32))
    BG_THRESHOLD = .3
    hard_neg_indices = tf.where(only_neg_maps > BG_THRESHOLD)

    #hard_neg_indices_mask = get_indicator_tensor(hard_neg_indices,tf.shape(labels_flat,out_type=tf.int64))
    hard_neg_indices_mask = tf.where(only_neg_maps>BG_THRESHOLD,tf.ones_like(labels_flat),tf.zeros_like(labels_flat))
    hard_neg_dist_mat = tf.multiply(dist_map_flat,tf.cast(hard_neg_indices_mask,tf.float32))

    values, selected_neg_indices = tf.nn.top_k(tf.transpose(hard_neg_dist_mat), bg_count)
    #selected_neg_indices = tf.squeeze(selected_neg_indices)
    selected_neg_indices = tf.transpose(selected_neg_indices)

    selected_pos_indices = tf.where(tf.equal(tf.squeeze(labels_flat),tf.constant(1)))
    selected_pos_indices = tf.cast(selected_pos_indices,tf.int32)
    #selected_pos_indices = tf.squeeze(selected_pos_indices)

    selected_indices = tf.concat(axis=0,values=[selected_pos_indices,selected_neg_indices])

    #dense_selected_mask = get_indicator_tensor(selected_indices,labels_flat.get_shape())
    #dense_selected_mask = tf.zeros(tf.shape(labels_flat))
    #dense_selected_mask = tf.constant(0.0,dtype = tf.float32,shape = tf.shape(labels_flat))#,verify_shape = False)
    #dense_selected_mask = tf.Variable(tf.identity(labels_flat),validate_shape = False)
    #ind_values = tf.ones_like(selected_indices)
    #tf.scatter_update(dense_selected_mask,selected_indices,ind_values)
    #dense_selected_mask = get_indicator_tensor(selected_indices,labels_flat.get_shape())


    return  selected_indices


def get_ohem_loss(loss_flat, labels_flat):

    #loss_flat = tf.squeeze(loss_flat)
    #labels_flat = tf.squeeze(labels_flat)
    print(loss_flat)
    print(labels_flat)

    total_fg_count = tf_count(labels_flat, 1)
    total_bg_count = tf_count(labels_flat, 0)


    bg_count = tf.minimum(total_bg_count,3*total_fg_count)

    labels_flat = tf.squeeze(labels_flat)
    pos_inds = tf.where(tf.equal(labels_flat, 1))
    neg_inds = tf.where(tf.equal(labels_flat, 0))

    loss_n_pos = tf.gather(loss_flat, pos_inds)
    loss_n_neg = tf.gather(loss_flat, neg_inds)
    print(loss_n_pos)

    loss_n_neg = tf.Print(loss_n_neg,[loss_n_neg,loss_n_pos])

    loss_n_neg, top_k_indices = tf.nn.top_k(tf.transpose(loss_n_neg), k=bg_count, sorted=False)
    loss_n_neg = tf.transpose(loss_n_neg)
    selected_loss = tf.concat(axis=0, values=(loss_n_pos, loss_n_neg))
    print(selected_loss)
    selected_loss = tf.expand_dims(selected_loss,axis = 1)
    #print(selected_loss)

    return  selected_loss



def dist_loss(logits, labels, dist_map=None):
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
        logits_flat = tf.reshape(logits, (-1, 2))
        # Weights flat
        dist_flat = tf.reshape(dist_map, (-1, 1))

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1,1))


        #mat = get_mulmat(logits_flat,label_flat,dist_flat)
        print (logits_flat)
        print(logits_flat.get_shape().ndims)
        print(label_flat.get_shape().ndims)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_flat, labels=tf.squeeze(label_flat), name='cross_entropy')
        #cross_entropy = tf.gather(cross_entropy,mat)
        cross_entropy = get_ohem_loss(cross_entropy,label_flat)


        #cross_entropy_mul = tf.mul(cross_entropy, tf.cast(mat,tf.float32), name='weighted_cross_entropy')
        # cross_entropy = -tf.mul(tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1]),weights_flat)
        #print(cross_entropy_mul)
        #cross_entropy_sum = tf.reduce_sum(cross_entropy_mul, reduction_indices=[1, 2], name='cross_entropy_sum')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        #print(cross_entropy_sum)
        print(cross_entropy_mean)
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss


def weighted_per_image_loss2(logits, labels, num_classes, weight_map=None):
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
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')

        print(cross_entropy)
        cross_entropy_mul = tf.multiply(cross_entropy, weight_map, name='weighted_cross_entropy')
        # cross_entropy = -tf.mul(tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1]),weights_flat)
        print(cross_entropy_mul)
        cross_entropy_sum = tf.reduce_sum(cross_entropy_mul, reduction_indices=[1, 2], name='cross_entropy_sum')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        print(cross_entropy_sum)
        print(cross_entropy_mean)
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss


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
        weights_flat = tf.reshape(weight_map, (-1, 1))

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        # cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax + epsilon), [1.0,1.0]), reduction_indices=[1])
        tmp1 = tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1])
        reshaped_tmp = tf.reshape(tmp1, (-1, 1))

        cross_entropy = -tf.multiply(reshaped_tmp, weights_flat)

        # cross_entropy = -tf.mul(tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1]),weights_flat)

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss


def softmax_accuracy():
    pass
