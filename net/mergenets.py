import tensorflow as tf
from net import segnet2
from net import osvos
#from lstm import md_lstm
from net import segnet_brn as segnet_brn


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

def conv_layer(inputT, shape, activation=True, name=None):
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
      kernel = segnet2._variable_with_weight_decay('ort_weights', shape=shape, initializer=segnet2.orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = segnet2._variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(bias)
      else:
        conv_out = bias
    return conv_out

"""
def inference_merge_two_branch_lstm(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:
        branch1 = tf.stop_gradient(branch1, name='merged_b1_sg')
        branch2 = tf.stop_gradient(branch2, name='merged_b2_sg')
        branch2 = segnet2.batch_norm_layer(branch2, phase_train, scope.name)

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        net, _ = md_lstm.multi_dimensional_rnn_while_loop(rnn_size=2, input_data=net, sh=[1, 1])



    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 2, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 2),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit
"""

def inference_merge_two_branch_bn2(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:
        branch1 = tf.stop_gradient(branch1, name='merged_b1_sg')
        branch2 = tf.stop_gradient(branch2, name='merged_b2_sg')
        branch2 = segnet2.batch_norm_layer(branch2, phase_train, scope.name)

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        net = tf.Print(net,[nb1,nb2])

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

def inference_merge_two_branch_bn(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:
        branch1 = tf.stop_gradient(branch1, name='merged_b1_sg')
        branch2 = tf.stop_gradient(branch2, name='merged_b2_sg')
        branch2 = segnet2.batch_norm_layer(branch2, phase_train, scope.name)

        branch1 = segnet2.conv_layer_with_bn(branch1, [1, 1, 64, 64], phase_train, True, name="merge_conv_b1")
        branch2 = segnet2.conv_layer_with_bn(branch2, [1, 1, 64, 64], phase_train, True, name="merge_conv_b2")

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        net = tf.Print(net,[nb1,nb2])

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
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        net = tf.Print(net,[nb1,nb2])
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

def inference_merge_two_branch_no_bn(images_branch1, phase_train):
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
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = conv_layer(net, [3, 3, 128, 64], False, name="merge_conv1")
        net = conv_layer(net, [3, 3, 64, 64], False, name="merge_conv2")
        net = conv_layer(net, [3, 3, 64, 32], False, name="merge_conv3")


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

def inference_merge_two_branch_baseline_brn(images_branch1, phase_train,return_branches = False):
    with tf.variable_scope('branch1') as scope:
        false_tensor = tf.constant(False)
        branch1 = segnet_brn.inference_encoder_decoder(images_branch1,false_tensor)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 128, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 128),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if not return_branches:
        return logit
    else:
        endpt = dict()
        endpt['branch1'] = branch1
        endpt['branch2'] = branch2
        return logit,endpt

def mini_encode_decode(net):
    batch_size = tf.shape(net)[0]
    IMG_HEIGHT = tf.shape(net)[1]  # IMAGE_HEIGHT
    IMG_WIDTH = tf.shape(net)[2]  # IMAGE_WIDTH

    net = conv_layer(net, [3, 3, 128, 64], True, name="encode_conv1_1")
    net = conv_layer(net, [3, 3, 64, 64], True, name="encode_conv1_2")
    net, pool1_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='auto_pool1')
    net = conv_layer(net, [3, 3, 64, 32], True, name="encode_conv2_1")
    net = conv_layer(net, [3, 3, 32, 32], True, name="encode_conv2_2")
    net, pool2_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                          padding='SAME', name='auto_pool1')

    out_size2 = tf.stack([batch_size,tf.cast(tf.ceil(IMG_HEIGHT / 2),tf.int32),tf.cast(tf.ceil(IMG_WIDTH / 2),tf.int32),tf.constant(32,tf.int32)])
    net = segnet_brn.deconv_layer(net, [2, 2, 32, 32],out_size2, 2, "up2")
    net = conv_layer(net, [3, 3, 32, 32], True, name="decode_conv2_1")
    net = conv_layer(net, [3, 3, 32, 32], True, name="decode_conv2_2")
    out_size1 = tf.stack([batch_size,IMG_HEIGHT,IMG_WIDTH ,32])
    net = segnet_brn.deconv_layer(net, [2, 2, 32, 32],out_size1, 2, "up1")
    net = conv_layer(net, [3, 3, 32, 32], True, name="decode_conv1_1")
    net = conv_layer(net, [3, 3, 32, 32], True, name="decode_conv1_2")
    return  net

def mini_encode_decode_v2(net,k=7):
    batch_size = tf.shape(net)[0]
    IMG_HEIGHT = tf.shape(net)[1]  # IMAGE_HEIGHT
    IMG_WIDTH = tf.shape(net)[2]  # IMAGE_WIDTH

    net = conv_layer(net, [k, k, 128, 64], True, name="encode_conv1_1")
    net = conv_layer(net, [k, k, 64, 64], True, name="encode_conv1_2")
    net, pool1_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name='auto_pool1')
    net = conv_layer(net, [k,k, 64, 32], True, name="encode_conv2_1")
    net = conv_layer(net, [k,k, 32, 32], True, name="encode_conv2_2")
    net, pool2_indices = tf.nn.max_pool_with_argmax(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                          padding='SAME', name='auto_pool1')

    out_size2 = tf.stack([batch_size,tf.cast(tf.ceil(IMG_HEIGHT / 2),tf.int32),tf.cast(tf.ceil(IMG_WIDTH / 2),tf.int32),tf.constant(32,tf.int32)])
    net = segnet_brn.deconv_layer(net, [2, 2, 32, 32],out_size2, 2, "up2")
    net = conv_layer(net, [k, k, 32, 32], True, name="decode_conv2_1")
    net = conv_layer(net, [k, k, 32, 32], True, name="decode_conv2_2")
    out_size1 = tf.stack([batch_size,IMG_HEIGHT,IMG_WIDTH ,32])
    net = segnet_brn.deconv_layer(net, [2, 2, 32, 32],out_size1, 2, "up1")
    net = conv_layer(net, [k, k, 32, 32], True, name="decode_conv1_1")
    net = conv_layer(net, [k, k, 32, 32], True, name="decode_conv1_2")
    return  net


def inference_merge_two_branch_brn_auto(images_branch1, phase_train,return_branches = False):
    return _inference_merge_two_branch_brn_auto(images_branch1, phase_train, k=3, return_branches=False)


def inference_merge_two_branch_brn_auto_k7(images_branch1, phase_train,return_branches = False):
    return  _inference_merge_two_branch_brn_auto(images_branch1, phase_train, k=7, return_branches=False)

def inference_merge_two_branch_brn_auto_k9(images_branch1, phase_train,return_branches = False):
    return  _inference_merge_two_branch_brn_auto(images_branch1, phase_train, k=9, return_branches=False)

def _inference_merge_two_branch_brn_auto(images_branch1, phase_train,k,return_branches = False):
    with tf.variable_scope('branch1') as scope:
        false_tensor = tf.constant(False)
        branch1 = segnet_brn.inference_encoder_decoder(images_branch1,false_tensor)

    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)


    with tf.variable_scope('merger') as scope:

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])

        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = mini_encode_decode_v2(net,k=k)


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 32, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 128),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if not return_branches:
        return logit
    else:
        endpt = dict()
        endpt['branch1'] = branch1
        endpt['branch2'] = branch2
        return logit,endpt


from lstm import rnn
def inference_merge_two_branch_brn_lstm(images_branch1, phase_train,return_branches = False):
    with tf.variable_scope('branch1') as scope:
        false_tensor = tf.constant(False)
        branch1 = segnet_brn.inference_encoder_decoder(images_branch1,false_tensor)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)



    with tf.variable_scope('merger') as scope:

        #net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])

        branch1 = tf.stop_gradient(branch1, name='merged_b1_sg')
        branch2 = tf.stop_gradient(branch2, name='merged_b2_sg')

        branch1 = conv_layer(branch1, [1, 1, 64, 32], True, name="branch1_dim_reduce1")
        branch2 = conv_layer(branch2, [1, 1, 64, 32], True, name="branch2_dim_reduce1")
        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')

        net = tf.Print(net,[nb1],message="LSTM step")
        netr1 = rnn.BiRowRNNStatic(net,n_hidden=16,num_features=64,scope_name="birow1")
        netc1 = rnn.BiColRNNStatic(net,n_hidden=16,num_features=64,scope_name="bicol1")
        net = tf.concat(axis=3, values=[netr1, netc1], name='mergelstm1')
        netr2 = rnn.BiRowRNNStatic(net,n_hidden=16,num_features=64,scope_name="birow2")
        netc2 = rnn.BiColRNNStatic(net,n_hidden=16,num_features=64,scope_name="bicol2")
        net = tf.concat(axis=3, values=[netr2, netc2], name='mergelstm2')
        net = tf.Print(net, [nb1], message="LSTM step complete")

    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 64, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 64),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if not return_branches:
        return logit
    else:
        endpt = dict()
        endpt['branch1'] = branch1
        endpt['branch2'] = branch2
        return logit,endpt


def inference_merge_psp_v4_brn(images_branch1, phase_train):
    print("mergenet - inference_merge_psp_v4_brn")
    with tf.variable_scope('branch1') as scope:
        false_tensor = tf.constant(False)
        #branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None, phase_train = phase_train)
        branch1 = segnet_brn.inference_encoder_decoder(images_branch1, false_tensor)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:




        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        netb1 = tf.stop_gradient(branch1, name='b1_sg')
        netb2 = tf.stop_gradient(branch2, name='b2_sg')


        netb1 = conv_layer(netb1, [3, 3, 64, 32], True, name="mergeb1_conv1")
        netb1 = conv_layer(netb1, [3, 3, 32, 16], True, name="mergeb1_conv2")
        netb1 = conv_layer(netb1, [3, 3, 16, 8], True, name="mergeb1_conv3")

        netb2 = conv_layer(netb2, [3, 3, 64, 32], True, name="mergeb2_conv1")
        netb2 = conv_layer(netb2, [3, 3, 32, 16], True, name="mergeb2_conv2")
        netb2 = conv_layer(netb2, [3, 3, 16, 8], True, name="mergeb2_conv3")

        pooled_b1 = pyramid_pool4(netb1,phase_train,name="ppb1")
        pooled_b2 = pyramid_pool4(netb2,phase_train,name="ppb2")

        net = tf.concat(axis=3, values=[netb1, netb2,pooled_b1,pooled_b2], name='merged_b1_b2_pool')

    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 22, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 22),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_two_branch_baseline(images_branch1, phase_train,return_branches = False):
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
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 128, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 128),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    if not return_branches:
        return logit
    else:
        endpt = dict()
        endpt['branch1'] = branch1
        endpt['branch2'] = branch2
        return logit,endpt

def inference_merge_two_branch_baseline_osvos_test(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        net = tf.concat(axis=3, values=[tf.zeros_like(branch1), branch2], name='merged_b1_b2')
        nb1 = tf.norm(branch1, ord=2)
        nb2 = tf.norm(branch2, ord=2)
        # find max as well
        net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 128, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 128),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit


def inference_merge_two_branch_l2normalized_fm(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        norm_branch1 = tf.nn.l2_normalize(branch1,dim=[1,2])
        norm_branch2 = tf.nn.l2_normalize(branch2,dim=[1,2])

        net = tf.concat(axis=3, values=[norm_branch1, norm_branch2], name='merged_b1_b2')
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

def inference_merge_two_branch_l2normalized_fm(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        norm_branch1 = tf.nn.l2_normalize(branch1,dim=[1,2])
        norm_branch2 = tf.nn.l2_normalize(branch2,dim=[1,2])

        net = tf.concat(axis=3, values=[norm_branch1, norm_branch2], name='merged_b1_b2')
        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
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

def inference_merge_two_branch_l2normalized_fm_no_bn(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        norm_branch1 = tf.nn.l2_normalize(branch1,dim=[1,2])
        norm_branch2 = tf.nn.l2_normalize(branch2,dim=[1,2])

        net = tf.concat(axis=3, values=[norm_branch1, norm_branch2], name='merged_b1_b2')
        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = conv_layer(net, [3, 3, 128, 64], False, name="merge_conv1")
        net = conv_layer(net, [3, 3, 64, 64], False, name="merge_conv2")
        net = conv_layer(net, [3, 3, 64, 32], False, name="merge_conv3")


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




def inference_merge_two_branch_rm_lstm(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None,phase_train=phase_train)
        branch1 = tf.nn.softmax(branch1)

    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:, :, :, ::-1]

        # images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2, _ = osvos.osvos(images_branch2)
            branch2 = tf.nn.sigmoid(branch2)

    with tf.variable_scope('merger') as scope:
        # nb1 = tf.norm(branch1, ord=2)
        # nb2 = tf.norm(branch2, ord=2)
        # find max as well
        # net = tf.Print(net,[nb1,nb2])
        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net, _ = md_lstm.multi_dimensional_rnn_while_loop(rnn_size=4, input_data=net, sh=[1, 1])


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                                     shape=[1, 1, 4, NUM_CLASSES],
                                                     initializer=segnet2.msra_initializer(1, 4),
                                                     wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_two_branch_rm_no_bn(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None,phase_train=phase_train)
        branch1 = tf.nn.softmax(branch1)

    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:, :, :, ::-1]

        # images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2, _ = osvos.osvos(images_branch2)
            branch2 = tf.nn.sigmoid(branch2)

    with tf.variable_scope('merger') as scope:

        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = conv_layer(net, [3, 3, 3, 3], False, name="merge_conv1")
        net = conv_layer(net, [3, 3, 3, 3], False, name="merge_conv2")
        net = conv_layer(net, [3, 3, 3, 3], False, name="merge_conv3")

    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                                     shape=[1, 1, 3, NUM_CLASSES],
                                                     initializer=segnet2.msra_initializer(1, 3),
                                                     wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_two_branch_rm(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None,phase_train=phase_train)
        branch1 = tf.nn.softmax(branch1)

    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:, :, :, ::-1]

        # images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2, _ = osvos.osvos(images_branch2)
            branch2 = tf.nn.sigmoid(branch2)

    with tf.variable_scope('merger') as scope:
        # nb1 = tf.norm(branch1, ord=2)
        # nb2 = tf.norm(branch2, ord=2)
        # find max as well
        # net = tf.Print(net,[nb1,nb2])
        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = segnet2.conv_layer_with_bn(net, [3, 3, 3, 3], phase_train, False, name="merge_conv1")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 3, 3], phase_train, False, name="merge_conv2")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 3, 3], phase_train, False, name="merge_conv3")

    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                                     shape=[1, 1, 3, NUM_CLASSES],
                                                     initializer=segnet2.msra_initializer(1, 3),
                                                     wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_two_branch_fm_multiplied(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)

    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos(images_branch2)

    with tf.variable_scope('merger') as scope:

        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.multiply(branch1, branch2, name='multiply_b1_b2')

        net = tf.stop_gradient(net, name='merged_b1_b2_sg')


        net = segnet2.conv_layer_with_bn(net, [3, 3, 32, 32], phase_train, False, name="merge_conv1")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 32, 32], phase_train, False, name="merge_conv2")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 32, 16], phase_train, False, name="merge_conv3")


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 16, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 16),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit


def inference_merge_two_branch_l2normalized_branch2(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos(images_branch2)

    with tf.variable_scope('merger') as scope:

        branch1 = tf.stop_gradient(branch1, name='b1_sg')
        branch2 = tf.stop_gradient(branch2, name='b2_sg')

        branch1 = segnet2.conv_layer_with_bn(branch1, [1, 1, 64, 64], phase_train, True, name="merge_conv_b1")
        branch2 = segnet2.conv_layer_with_bn(branch2, [1, 1, 1, 2], phase_train, True, name="merge_conv_b2")


        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')


        net = segnet2.conv_layer_with_bn(net, [3, 3, 66, 64], phase_train, False, name="merge_conv1")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 64, 32], phase_train, False, name="merge_conv2")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 32, 16], phase_train, False, name="merge_conv3")


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 16, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 16),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_two_branch_l2normalized_branch(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos(images_branch2)

    with tf.variable_scope('merger') as scope:

        norm_branch1 = tf.nn.l2_normalize(branch1,dim=[1,2,3])
        norm_branch2 = tf.nn.l2_normalize(branch2,dim=[1,2,3])

        net = tf.concat(axis=3, values=[norm_branch1, norm_branch2], name='merged_b1_b2')
        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        net = segnet2.conv_layer_with_bn(net, [3, 3, 65, 32], phase_train, False, name="merge_conv1")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 32, 32], phase_train, False, name="merge_conv2")
        net = segnet2.conv_layer_with_bn(net, [3, 3, 32, 16], phase_train, False, name="merge_conv3")


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 16, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 16),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit


def inference_merge_two_branch_fm_dropout_no_bn(images_branch1, phase_train):
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
        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        def mask_x_1():
            return tf.identity(tf.concat([tf.ones_like(branch1), tf.ones_like(branch2)], axis=3))

        def mask_x_2():
            return tf.identity(tf.concat([tf.ones_like(branch1) * 2, tf.zeros_like(branch2)], axis=3))

        def mask_x_3():
            return tf.identity(tf.concat([tf.zeros_like(branch1), tf.ones_like(branch2) * 2], axis=3))

        gen = tf.random_uniform([1])
        gen = tf.Print(gen, [gen])
        drop_mask = tf.cond(tf.logical_and(phase_train,gen[0] > 0.5), lambda: tf.cond(gen[0] < 0.75, mask_x_2, mask_x_3), mask_x_1)
        net = tf.multiply(net,drop_mask,name="mask_mul")

        net = conv_layer(net, [3, 3, 128, 64], False, name="merge_conv1")
        net = conv_layer(net, [3, 3, 64, 64], False, name="merge_conv2")
        net = conv_layer(net, [3, 3, 64, 32], False, name="merge_conv3")


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


def inference_merge_two_branch_l2normalized(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        branch1 = segnet2.inference_encoder_decoder(images_branch1,phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        norm_branch1 = tf.nn.l2_normalize(branch1,dim=3)
        norm_branch2 = tf.nn.l2_normalize(branch2,dim=3)

        net = tf.concat(axis=3, values=[norm_branch1, norm_branch2], name='merged_b1_b2')
        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
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


def pyramid_pool(input,phase_train):

    with tf.variable_scope('pyramid_pool') as scope:
        p1 = 30
        p2 = 60
        p3 = 120

        inp_shape = tf.shape(input)

        pool1 = tf.nn.avg_pool(input, [1, p1, p1, 1], [1, p1, p1, 1], padding='SAME')
        pool2 = tf.nn.avg_pool(input, [1, p2, p2, 1], [1, p2, p2, 1], padding='SAME')
        pool3 = tf.nn.avg_pool(input, [1, p3, p3, 1], [1, p3, p3, 1], padding='SAME')

        pool1_resized = tf.image.resize_bilinear(pool1,inp_shape[1:3])
        pool2_resized = tf.image.resize_bilinear(pool2,inp_shape[1:3])
        pool3_resized = tf.image.resize_bilinear(pool3,inp_shape[1:3])

        pool1_reduced = conv_layer(pool1_resized, [1, 1, 128, 1], True, name="pool1_dim_red")
        pool2_reduced = conv_layer(pool2_resized, [1, 1, 128, 1], True, name="pool2_dim_red")
        pool3_reduced = conv_layer(pool3_resized, [1, 1, 128, 1], True, name="pool3_dim_red")

        concat = tf.concat([pool1_reduced, pool2_reduced, pool3_reduced], axis=3)


    return concat

def pyramid_pool3(input,phase_train):

    with tf.variable_scope('pyramid_pool') as scope:
        p1 = 30
        p2 = 60
        p3 = 120

        inp_shape = tf.shape(input)

        pool1 = tf.nn.avg_pool(input, [1, p1, p1, 1], [1, p1, p1, 1], padding='SAME')
        pool2 = tf.nn.avg_pool(input, [1, p2, p2, 1], [1, p2, p2, 1], padding='SAME')
        pool3 = tf.nn.avg_pool(input, [1, p3, p3, 1], [1, p3, p3, 1], padding='SAME')

        pool1_resized = tf.image.resize_bilinear(pool1,inp_shape[1:3])
        pool2_resized = tf.image.resize_bilinear(pool2,inp_shape[1:3])
        pool3_resized = tf.image.resize_bilinear(pool3,inp_shape[1:3])

        pool1_reduced = conv_layer(pool1_resized, [1, 1, 16, 1], False, name="pool1_dim_red")
        pool2_reduced = conv_layer(pool2_resized, [1, 1, 16, 1], False, name="pool2_dim_red")
        pool3_reduced = conv_layer(pool3_resized, [1, 1, 16, 1], False, name="pool3_dim_red")

        concat = tf.concat([pool1_reduced, pool2_reduced, pool3_reduced], axis=3)


    return concat

def pyramid_pool4(input,phase_train,name):

    with tf.variable_scope('pyramid_pool') as scope:
        p1 = 30
        p2 = 60
        p3 = 120

        inp_shape = tf.shape(input)

        pool1 = tf.nn.avg_pool(input, [1, p1, p1, 1], [1, p1, p1, 1], padding='SAME')
        pool2 = tf.nn.avg_pool(input, [1, p2, p2, 1], [1, p2, p2, 1], padding='SAME')
        pool3 = tf.nn.avg_pool(input, [1, p3, p3, 1], [1, p3, p3, 1], padding='SAME')

        pool1_resized = tf.image.resize_bilinear(pool1,inp_shape[1:3])
        pool2_resized = tf.image.resize_bilinear(pool2,inp_shape[1:3])
        pool3_resized = tf.image.resize_bilinear(pool3,inp_shape[1:3])

        pool1_reduced = conv_layer(pool1_resized, [1, 1, 8, 1], True, name=name+"pool1_dim_red")
        pool2_reduced = conv_layer(pool2_resized, [1, 1, 8, 1], True, name=name+"pool2_dim_red")
        pool3_reduced = conv_layer(pool3_resized, [1, 1, 8, 1], True, name=name+"pool3_dim_red")

        concat = tf.concat([pool1_reduced, pool2_reduced, pool3_reduced], axis=3)


    return concat

def inference_merge_psp_v4(images_branch1, phase_train):
    print("mergenet - inference_merge_psp_v4")
    with tf.variable_scope('branch1') as scope:
        #branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None, phase_train = phase_train)
        branch1 = segnet2.inference_encoder_decoder(images_branch1, phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:




        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        netb1 = tf.stop_gradient(branch1, name='b1_sg')
        netb2 = tf.stop_gradient(branch2, name='b2_sg')


        netb1 = conv_layer(netb1, [3, 3, 64, 32], True, name="mergeb1_conv1")
        netb1 = conv_layer(netb1, [3, 3, 32, 16], True, name="mergeb1_conv2")
        netb1 = conv_layer(netb1, [3, 3, 16, 8], True, name="mergeb1_conv3")

        netb2 = conv_layer(netb2, [3, 3, 64, 32], True, name="mergeb2_conv1")
        netb2 = conv_layer(netb2, [3, 3, 32, 16], True, name="mergeb2_conv2")
        netb2 = conv_layer(netb2, [3, 3, 16, 8], True, name="mergeb2_conv3")

        pooled_b1 = pyramid_pool4(netb1,phase_train,name="ppb1")
        pooled_b2 = pyramid_pool4(netb2,phase_train,name="ppb2")

        net = tf.concat(axis=3, values=[netb1, netb2,pooled_b1,pooled_b2], name='merged_b1_b2_pool')

    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 22, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 22),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_psp_v3(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        #branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None, phase_train = phase_train)
        branch1 = segnet2.inference_encoder_decoder(images_branch1, phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:


        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')

        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')


        net = conv_layer(net, [3, 3, 128, 64], False, name="merge_conv1")
        net = conv_layer(net, [3, 3, 64, 32], False, name="merge_conv2")
        net = conv_layer(net, [3, 3, 32, 16], False, name="merge_conv3")
        pooled_out = pyramid_pool3(net,phase_train)

        net = tf.concat(axis=3, values=[pooled_out, net], name='merged_b1_b2')


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 19, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 19),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_psp_v2(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        #branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None, phase_train = phase_train)
        branch1 = segnet2.inference_encoder_decoder(images_branch1, phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:


        net = tf.concat(axis=3, values=[branch1, branch2], name='merged_b1_b2')

        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        pooled_out = pyramid_pool(net,phase_train)

        net = conv_layer(net, [3, 3, 128, 64], False, name="merge_conv1")
        net = conv_layer(net, [3, 3, 64, 32], False, name="merge_conv2")
        net = conv_layer(net, [3, 3, 32, 16], False, name="merge_conv3")
        net = tf.concat(axis=3, values=[pooled_out, net], name='merged_b1_b2')


    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
        kernel = segnet2._variable_with_weight_decay('weights',
                                             shape=[1, 1, 19, NUM_CLASSES],
                                             initializer=segnet2.msra_initializer(1, 19),
                                             wd=0.0005)
        conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
        biases = segnet2._variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    return logit

def inference_merge_psp_v1(images_branch1, phase_train):
    with tf.variable_scope('branch1') as scope:
        #branch1 = segnet2.inference_vgg16_withskip(images_branch1, labels=None, phase_train = phase_train)
        branch1 = segnet2.inference_encoder_decoder(images_branch1, phase_train)
    with tf.variable_scope('branch2') as scope:
        images_branch2 = images_branch1[:, :, :, 0:3]
        images_branch2 = images_branch2[:,:,:,::-1]

        #images_branch2 = tf.placeholder(tf.float32, [2, None, None, 3])

        with slim.arg_scope(osvos.osvos_arg_scope()):
            branch2,_ = osvos.osvos_net(images_branch2)

    with tf.variable_scope('merger') as scope:

        norm_branch1 = tf.nn.l2_normalize(branch1,dim=[1,2])
        norm_branch2 = tf.nn.l2_normalize(branch2,dim=[1,2])

        net = tf.concat(axis=3, values=[norm_branch1, norm_branch2], name='merged_b1_b2')

        #nb1 = tf.norm(branch1, ord=2)
        #nb2 = tf.norm(branch2, ord=2)
        # find max as well
        #net = tf.Print(net,[nb1,nb2])
        net = tf.stop_gradient(net, name='merged_b1_b2_sg')

        pooled_out = pyramid_pool(net,phase_train)
        net = tf.concat(axis=3, values=[pooled_out, net], name='merged_b1_b2')

        net = segnet2.conv_layer_with_bn(net, [3, 3, 131, 64], phase_train, False, name="merge_conv1")
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

