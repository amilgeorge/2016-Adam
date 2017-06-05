'''
Created on Dec 1, 2016

@author: george
'''
# -*- coding: utf-8 -*-
from net.segnet2 import NUM_CLASSES
import tensorflow as tf
import numpy as np
from dataprovider import imdbdataprovider
from dataprovider.preprocess import vgg_preprocess, reverse_vgg_preprocess
from dataprovider.imdbdataprovider import InputProvider
from dataprovider import frame_no_calculator as fnc
from common.logger import getLogger
from common.diskutils import ensure_dir
from net import segnet2 as segnet
import time
import os
import test_merge_osvos_net
import test_merge_osvos_net_sigmoid
from net import mergenets
from net import osvos
from ops.segnet_loss import weighted_per_image_loss2
from dataprovider import imdb
#import tracemalloc
#import gc

slim = tf.contrib.slim

OFFSETS = [(fnc.POLICY_OFFSET,1)]

CHECKPOINT_BRANCH1 = 'exp/s480pvgg-davis2016-O1-osvosold-reg1e-4-mo<1e-2>-de-1/iters-500000'

dbname = imdb.IMDB_SEQ_DAVIS2016

RUN_ID = "checksanity-{}-B1O{}--momentum-<1e-11>-withsg-4".format(dbname,OFFSETS[0][1])


EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

IMG_HEIGHT = 480
IMG_WIDTH = 854

all_train_seqs = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'dog-agility',
            'drift-turn', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia',
            'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding', 'rhino', 'rollerblade',
            'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

all_test_seqs  = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl',
             'dog','drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump',
             'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

MAX_ITER_PER_SEQ = 250


def reinit_branch_and_fuse(session,seq):
    reinit_branch(session,seq)
    reinit_fusion(session,seq)

def reinit_fusion(session,seq):
    checkpoint_file = os.path.join('..','OSVOS-TensorFlow','models',seq,'{}.ckpt-2000'.format(seq))
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    orig_fuse_tensor = reader.get_tensor('osvos/upscore-fuse/weights')
    orig_fuse_biases = reader.get_tensor('osvos/upscore-fuse/biases')
    branch1_fuse = np.zeros((1,1,64,1),dtype=np.float32)
    concat = np.concatenate((branch1_fuse,orig_fuse_tensor),axis=2)
    vars_corresp = dict()
    vars_corresp['merger/upscore-fuse/weights'] = concat
    vars_corresp['merger/upscore-fuse/biases'] = orig_fuse_biases
    assign_func = slim.assign_from_values_fn(vars_corresp)
    assign_func(session)

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

def reinit_branch(session,seq):
    """
    Re init osvos branch with fine tuned checkpoint for sequence
    :param session:
    :param seq:
    :return:
    """

    checkpoint_file = os.path.join('..','OSVOS-TensorFlow','models',seq,'{}.ckpt-2000'.format(seq))
    mergenets.initialize_merge_net(session, None, checkpoint_file)



if __name__ == '__main__':

    #tracemalloc.start()

    #gc.enable()
    #gc.set_debug(gc.DEBUG_LEAK)

    #reinit_fusion(None,'bear')
    ensure_dir(EXP_DIR)
    ensure_dir(LOGS_DIR)        
    logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))

    RNG_SEED = 3
    np.random.seed(RNG_SEED)
    pfc = fnc.get(OFFSETS[0][0], OFFSETS[0][1])

    with tf.Graph().as_default():
        tf.set_random_seed(1)

        NUM_CLASSES = segnet.NUM_CLASSES
        # Create placeholders for input and output
        inp_branch1 = tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,7],name='input_branch1')
        #inp_branch2 = tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,7],name='input_branch2')

        label =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,1],name='label')
        weights =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,1],name='weights')
        is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")

        label_int = tf.cast(label, tf.int32)
        logit = mergenets.inference_merge_net_sanity_check1(inp_branch1, is_training_pl)
        loss = osvos.class_balanced_cross_entropy_loss(logit, label)

        #out = tf.nn.softmax(logit)
        out = tf.nn.sigmoid(logit)
        predictions = tf.greater_equal(out, 0.5, "predictions")
        predictions_int = tf.cast(predictions,tf.int64)

        net_dict = {"inp_pl": inp_branch1,
                    "label_pl": label,
                    "out": out,
                    "is_training_pl": is_training_pl,
                    "prediction":predictions
                    }

        # Declare the optimizer
        global_step_var = tf.Variable(0, trainable=False)
            
        learning_rate = tf.train.exponential_decay(0.01, global_step_var, 10,
                                       0.1, staircase=True)

        logger.info('using Momentum optimizer')
        optimizer = tf.train.MomentumOptimizer(1e-11,0.9)
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='')
        gradients = optimizer.compute_gradients(loss)

        apply_gradient_op = optimizer.apply_gradients(gradients, global_step_var)
        batch_size = 1

        ##### Summaries #######
        prediction_reshaped = tf.reshape(predictions_int, [-1])
        label_reshaped=tf.reshape(tf.cast(label, tf.int64), [-1])

        acc = tf.contrib.metrics.accuracy(prediction_reshaped, label_reshaped)
        confusion_matrix = tf.contrib.metrics.confusion_matrix(prediction_reshaped,label_reshaped,num_classes=2)

        #regularization_loss = tf.contrib.losses.get_regularization_losses()

        tp_tensor = confusion_matrix[1,1]
        fp_tensor = confusion_matrix[1,0]
        fn_tensor = confusion_matrix[0,1]

        precision = tp_tensor/(tp_tensor + fp_tensor)
        recall = tp_tensor/(tp_tensor + fn_tensor)
        jaccard = tp_tensor/(tp_tensor + fn_tensor + fp_tensor)

        out_reshaped = tf.reshape(out,[-1,IMG_HEIGHT,IMG_WIDTH,NUM_CLASSES])

        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        tf.summary.histogram(logit.op.name + '/val',logit)
                
        tf.summary.scalar('/loss', loss)
        tf.summary.scalar('/accuracy',acc)
        tf.summary.scalar('/precision',precision)
        tf.summary.scalar('/recall',recall)
        tf.summary.scalar('/jaccard',jaccard)
        merged_summary = tf.summary.merge_all()
        ########################

        # Input Provider
        inp_provider_dict = {}
        for seq in all_train_seqs:
            sub_db_name = '{},{}'.format(imdb.IMDB_FINETUNE_DAVIS2016,seq)
            inp_provider = InputProvider(db_name = sub_db_name,prev_frame_calculator=pfc)
            inp_provider_dict[seq] = inp_provider

        saver = tf.train.Saver(max_to_keep = 10)
        #saver.export_meta_graph("metagraph.meta", as_text=True)


        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

            checkpoint_file = tf.train.latest_checkpoint(EXP_DIR)

            if checkpoint_file:
                logger.info('using checkpoint :{}'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            else :   
                # Build an initialization operation to run below.
                init = tf.global_variables_initializer()

                # Start running operations on the Graph.
                sess.run(init)

                mergenets.initialize_merge_net(sess,CHECKPOINT_BRANCH1,None)


            train_summary_writer = tf.summary.FileWriter(EVENTS_DIR + '/train', sess.graph)

            # test all sequences without training
            if 0:
                for seq in all_test_seqs+all_train_seqs:
                    curr_seq = seq
                    reinit_branch_and_fuse(sess, curr_seq)
                    test_merge_osvos_net_sigmoid.test_sequence(sess, net_dict,curr_seq, "../Results/test_merge_sanity", pfc)


            max_iters = 20000
            curr_seq = 'bmx-bumps'
            input_provider = inp_provider_dict[curr_seq]
            input_provider.initialize_iterator(batch_size)
            reinit_branch_and_fuse(sess, curr_seq)
            tmp_dir = 'davis_imdb_samp_off_cs'
            ensure_dir(tmp_dir)

            while global_step_var.eval() <= max_iters:

                # select input provider

                step = global_step_var.eval()
                if step % 100 == 0:
                    test_out_dir = os.path.join('test_out', RUN_ID, 'iter-{}'.format(step))
                    ensure_dir(test_out_dir)
                    test_merge_osvos_net_sigmoid.test_sequence(sess, net_dict,curr_seq, test_out_dir, pfc)

                sequence_batch = input_provider.next_mini_batch()

                #tmp_img_path = os.path.join(tmp_dir,'{}.png'.format(step))
                #imdbdataprovider._debug(reverse_vgg_preprocess(sequence_batch.images)[0,:,:,:],
                #                        sequence_batch.labels[0,:,:],tmp_img_path)




                result = sess.run([apply_gradient_op, loss,merged_summary,acc,precision,recall,jaccard,tp_tensor,
                                   fn_tensor,fp_tensor,confusion_matrix],
                                  feed_dict={inp_branch1:sequence_batch.images,
                                            label:preprocess_labels(sequence_batch.labels),
                                            is_training_pl:True})

                loss_value = result[1]
                acc_value = result[3]
                precision_value = result[4]
                recall_value = result[5]
                jaccard_value = result[6]

                logger.info('seq:{} iters:{}, loss :{} accuracy:{} p:{} r:{} j:{}'.format(curr_seq,step
                                               ,loss_value,acc_value,precision_value,recall_value,jaccard_value))

                if step%100 ==0:
                    #import pdb
                    #pdb.set_trace()
                    train_summary_writer.add_summary(result[2], step )

                if step % 100 == 0:
                    logger.info('Saving weights.')
                    saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                    logger.info('Flushing .')
                    train_summary_writer.flush()



                
            train_summary_writer.close()


    
    
