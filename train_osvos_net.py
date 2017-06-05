'''
Created on Dec 1, 2016

@author: george
'''
# -*- coding: utf-8 -*-
from net.segnet2 import NUM_CLASSES
import tensorflow as tf
import numpy as np
from dataprovider.rawimdbdataprovider import InputProvider
from dataprovider import frame_no_calculator as fnc

from common.logger import getLogger
from common.diskutils import ensure_dir
from net import osvos
import time
import os
import test_segnet2
from dataprovider import imdb
import sys
from datetime import datetime
import test_osvos_net
#import tracemalloc
#import gc

slim = tf.contrib.slim


OFFSETS = [(fnc.POLICY_OFFSET,1)]#list(range(1,7))#[10]
#OFFSETS = [(fnc.POLICY_DEFAULT_ZERO,0)]
offset_string = "-".join(str(x) for p,x in OFFSETS)
dbname = imdb.IMDB_DAVIS_2016
lr = 5e-6
RUN_ID = "osvos-{}-O{}-de-1-osvosckpt-lr-10-1".format(dbname,offset_string)

CHECKPOINT = None#'exp/segnetvggwithskip-half-wl-osvos-O10-1/iters-45000'

EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)
ensure_dir(LOGS_DIR)
logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))

#
# def dump_garbage():
#     """
#     show us what's the garbage about
#     """
#
#     # force collection
#     print ("\nGARBAGE:")
#     gc.collect()
#
#     print ("\nGARBAGE OBJECTS:")
#     for x in gc.garbage:
#         s = str(x)
#         if len(s) > 80: s = s[:80]
#         print (type(x), "\n  ", s)


# TO DO: Move preprocessing into Tensorflow



# TO DO: Move preprocessing into Tensorflow

def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    #max_mask = np.max(label) * 0.5
    #print(max_mask)
    #label = np.greater(label, max_mask)
    label = np.expand_dims(label, axis=3)
    #print (label.shape)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return label

def _train(input_provider,initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False, config=None, finetune=1,
           test_image_path=None, ckpt_name="osvos",pfc=None ):
    """Train OSVOS
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: Size of the training batch
    momentum: Value of the momentum parameter for the Momentum optimizer
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning
    test_image_path: If image path provided, every save_step the result of the network with this image is stored
    Returns:
    """
    model_name = os.path.join(logs_path, ckpt_name+".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 7])
    input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])

    # Create the network
    with slim.arg_scope(osvos.osvos_arg_scope()):
        net, end_points = osvos.osvos(input_image)

    # Initialize weights from pre-trained model
    if finetune == 0:
        #init_weights = osvos.load_vgg_imagenet(initial_ckpt)
        #init_conv1 = osvos.load_first_conv(initial_ckpt)
        init_weights = osvos.load_ckpt_ch7(initial_ckpt)
        init_conv1 = osvos.load_first_convV2(initial_ckpt)

    net_dict = {}
    out = tf.nn.sigmoid(net)
    net_dict['inp_pl'] = input_image
    net_dict['out'] = out

    # Define loss
    with tf.name_scope('losses'):
        if supervison == 1 or supervison == 2:
            dsn_2_loss = osvos.class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_2-cr'], input_label)
            tf.summary.scalar('dsn_2_loss', dsn_2_loss)
            dsn_3_loss = osvos.class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_3-cr'], input_label)
            tf.summary.scalar('dsn_3_loss', dsn_3_loss)
            dsn_4_loss = osvos.class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_4-cr'], input_label)
            tf.summary.scalar('dsn_4_loss', dsn_4_loss)
            dsn_5_loss = osvos.class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_5-cr'], input_label)
            tf.summary.scalar('dsn_5_loss', dsn_5_loss)

        main_loss = osvos.class_balanced_cross_entropy_loss(net, input_label)
        tf.summary.scalar('main_loss', main_loss)

        if supervison == 1:
            output_loss = dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss + main_loss
        elif supervison == 2:
            output_loss = 0.5 * dsn_2_loss + 0.5 * dsn_3_loss + 0.5 * dsn_4_loss + 0.5 * dsn_5_loss + main_loss
        elif supervison == 3:
            output_loss = main_loss
        else:
            sys.exit('Incorrect supervision id, select 1 for supervision of the side outputs, 2 for weak supervision '
                     'of the side outputs and 3 for no supervision of the side outputs')
        total_loss = output_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = {}
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator[ind] = tf.ConditionalAccumulator(grads_and_vars[ind][0].dtype)
        with tf.name_scope('apply_gradient'):
            layer_lr = osvos.parameter_lr()
            grad_accumulator_ops = []
            for var_ind, grad_acc in grad_accumulator.items():
                var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                var_grad = grads_and_vars[var_ind][0]
                grad_accumulator_ops.append(grad_acc.apply_grad(var_grad * layer_lr[var_name],
                                                                local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in grad_accumulator.items():
                mean_grads_and_vars.append(
                    (grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)

        # add summary
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Log training info
    merged_summary_op = tf.summary.merge_all()

    # Log evolution of test image
    if test_image_path is not None:
        probabilities = tf.nn.sigmoid(net)
        img_summary = tf.summary.image("Output probabilities", probabilities, max_outputs=1)
    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # Option in the session options=run_options
    # run_metadata = tf.RunMetadata() # Option in the session run_metadata=run_metadata
    # summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
    with tf.Session(config=config) as sess:
        print ('Init variable')
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            # Load pre-trained model
            if finetune == 0:
                print('Initializing from pre-trained imagenet model...')
                init_weights(sess)
                init_conv1(sess)
            else:
                print('Initializing from specified pre-trained model...')
                # init_weights(sess)
                var_list = []
                for var in tf.global_variables():
                    var_type = var.name.split('/')[-1]
                    if 'weights' in var_type or 'bias' in var_type:
                        var_list.append(var)
                saver_res = tf.train.Saver(var_list=var_list)
                saver_res.restore(sess, initial_ckpt)
            step = 1
        sess.run(osvos.interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print ('Start training')
        while step < max_training_iters + 1:
            # Average the gradient
            batch_losses = []
            for _ in range(0, iter_mean_grad):
                dataminibatch = input_provider.next_mini_batch()
                mb_imgs = osvos.preprocess_img_ch7(dataminibatch.images)
                mb_labels = preprocess_labels(dataminibatch.labels)

                run_res = sess.run([total_loss, merged_summary_op] + grad_accumulator_ops,
                                   feed_dict={input_image: mb_imgs, input_label: mb_labels})
                batch_losses.append(run_res[0])
                summary = run_res[1]
                logger.info(" loss = {:.4f}".format(run_res[0]))

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                logger.info ("{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, np.mean(batch_losses)))

            # Save a checkpoint
            if step % save_step == 0:
                test_out_dir = os.path.join('test_out', RUN_ID, 'iter-{}'.format(step))
                ensure_dir(test_out_dir)
                save_path = saver.save(sess, model_name, global_step=global_step)
                test_osvos_net.test_network(sess, net_dict, test_out_dir, pfc)
                logger.info ("Model saved in file: %s" % save_path)

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            logger.info ("Model saved in file: %s" % save_path)

            logger.info('Finished training.')

def train_p(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                 display_step, global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False,
                 config=None, test_image_path=None, ckpt_name="osvos",pfc=None):
    """Train OSVOS parent network
    Args:
    See _train()
    Returns:
    """
    finetune = 0
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, test_image_path,
           ckpt_name,pfc)

if __name__ == '__main__':
    # User defined parameters
    gpu_id = 0
    osvos_root = os.path.join('..','OSVOS-TensorFlow')
    # Training parameters
    imagenet_ckpt = os.path.join(osvos_root,'models/OSVOS_parent/OSVOS_parent.ckpt-50000')
    logs_path = os.path.join( 'exp_osvos', RUN_ID)
    iter_mean_grad = 10
    max_training_iters_1 = 15000
    max_training_iters_2 = 30000
    max_training_iters_3 = 50000
    save_step = 1000
    display_step = 1
    ini_learning_rate = 1e-10
    boundaries = [10000, 15000, 25000, 30000, 40000]
    values = [ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate,
              ini_learning_rate * 0.1]

    # Define Dataset
    pfc = fnc.get(OFFSETS[0][0], OFFSETS[0][1])
    input_provider = InputProvider(db_name = dbname,prev_frame_calculator=pfc)
    input_provider.initialize_iterator(1)

    ensure_dir(logs_path)

    # Train the network
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_p(input_provider, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters_1, save_step,
                               display_step, global_step, iter_mean_grad=iter_mean_grad, test_image_path=None,
                               ckpt_name='OSVOS_p',pfc=pfc)

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(max_training_iters_1, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_p(input_provider, imagenet_ckpt, 2, learning_rate, logs_path, max_training_iters_2, save_step,
                               display_step, global_step, iter_mean_grad=iter_mean_grad, resume_training=True,
                               test_image_path=None, ckpt_name='OSVOS_p',pfc=pfc)

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(max_training_iters_2, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_p(input_provider, imagenet_ckpt, 3, learning_rate, logs_path, max_training_iters_3, save_step,
                               display_step, global_step, iter_mean_grad=iter_mean_grad, resume_training=True,
                               test_image_path=None, ckpt_name='OSVOS_p',pfc=pfc)

    
    
    
