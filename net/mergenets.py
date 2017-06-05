import tensorflow as tf
from net import segnet2
from net import osvos

NUM_CLASSES = 2
slim = tf.contrib.slim

def add_loss_summaries(total_loss):
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
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def inference_merge_two_branch(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = segnet2.conv_layer_with_bn(net, [3, 3, 128, 64], phase_train, False, name="merge_conv1")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 64, 64], phase_train, False, name="merge_conv2")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 64, 32], phase_train, False, name="merge_conv3")


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 32, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 32),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit


def inference_merge_net_sanity_check1(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            osvosnet, osvosnet_end_points = osvos.osvos(images_branch2)
            branch2 = osvosnet_end_points['osvos/concat_side']


    with tf.variable_scope('merger') as scope:

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')
        with slim.arg_scope(osvos.osvos_arg_scope()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],padding='SAME'):
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None):
                    net = slim.conv2d(net, 1, [1, 1], scope='upscore-fuse')


    return net


def initialize_merge_net(sess,checkpoint_branch1,checkpoint_branch2):
    all_vars = tf.global_variables()
    print (all_vars)
    if not checkpoint_branch1 is None:
        branch1_vars = {v.op.name.replace("branch1/",""):v for v in all_vars if v.name.startswith("branch1")}
        saver_branch1 = tf.train.Saver(branch1_vars)
        saver_branch1.restore(sess, checkpoint_branch1)

    if  not checkpoint_branch2 is None:
        branch2_vars = {v.op.name.replace("branch2/",""):v for v in all_vars if v.name.startswith("branch2")}
        saver_branch2 = tf.train.Saver(branch2_vars)
        saver_branch2.restore(sess, checkpoint_branch2)

