'''
Created on Dec 1, 2016

@author: george
'''
# -*- coding: utf-8 -*-
from net.segnet2 import NUM_CLASSES
import tensorflow as tf
import numpy as np
from dataprovider.imdbdataprovider import InputProvider
from dataprovider import frame_no_calculator as fnc
from common.logger import getLogger
from common.diskutils import ensure_dir
from net import segnet2 as segnet
import time
import os
import test_merge_osvos_net
from net import mergenets
from ops.segnet_loss import weighted_per_image_loss2
from dataprovider import imdb
#import tracemalloc
#import gc

slim = tf.contrib.slim

OFFSETS = [(fnc.POLICY_OFFSET,1)]

#CHECKPOINT_BRANCH1 = 'exp/s480pvgg-davis2016-O1-osvosold-reg1e-4-mo<1e-2>-de-1/iters-500000'
CHECKPOINT_BRANCH1 = 'exp/s480pvgg-davis2016-O1-osvosold-reg1e-4-adam<1e-6>-de-1/iters-500000'


dbname = imdb.IMDB_SEQ_DAVIS2016

RUN_ID = "mergeosvosnet-{}-B1O-{}-adam<1e-6>-opt-adam-<1e-4>-3".format(dbname,OFFSETS[0][1])


EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

IMG_HEIGHT = 480
IMG_WIDTH = 854

all_seqs = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'dog-agility',
            'drift-turn', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia',
            'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding', 'rhino', 'rollerblade',
            'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

MAX_ITER_PER_SEQ = 250


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

        label =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH],name='label')
        weights =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH],name='weights')
        is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")

        label_int = tf.cast(label, tf.int32)
        logit = mergenets.inference_merge_two_branch(inp_branch1, is_training_pl)
        loss = weighted_per_image_loss2(logit, label_int, num_classes=NUM_CLASSES, weight_map=weights)
        loss_averages_op = mergenets.add_loss_summaries(loss)

        logit = tf.reshape(logit, (-1, NUM_CLASSES))
        out=tf.reshape(tf.nn.softmax(logit),[-1,IMG_HEIGHT,IMG_WIDTH,2])

        net_dict = {"inp_pl": inp_branch1,
                   "label_pl": label,
                   "out": out,
                   "is_training_pl": is_training_pl,
                   "loss": loss}

        #out = tf.nn.softmax(logit)
        predictions = tf.arg_max(out, 3, "predictions")

        # Declare the optimizer
        global_step_var = tf.Variable(1, trainable=False)
            
        learning_rate = tf.train.exponential_decay(0.01, global_step_var, 10,
                                       0.1, staircase=True)

        with tf.control_dependencies([loss_averages_op]):
        #optimizer = tf.train.GradientDescentOptimizer(0.01)
            logger.info('using adam optimizer')
            optimizer = tf.train.AdamOptimizer(1e-4)
            #optimizer = tf.train.MomentumOptimizer(0.001,0.9)

            gradients = optimizer.compute_gradients(loss)
            
        
                    
            
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step_var)
        #apply_gradient_op = segnet.train(loss, global_step_var)    
        max_iters = 250000
        batch_size = segnet.BATCH_SIZE
            
        # Input Provider
        inp_provider_dict = {}
        for seq in all_seqs:
            sub_db_name = '{},{}'.format(imdb.IMDB_SEQ_DAVIS2016,seq)
            inp_provider = InputProvider(db_name = sub_db_name,prev_frame_calculator=pfc)
            inp_provider_dict[seq] = inp_provider


    
        ##### Summaries #######
        prediction_reshaped = tf.reshape(predictions, [-1])
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
                
        tf.summary.scalar('/loss', loss)
        tf.summary.scalar('/accuracy',acc)
        tf.summary.scalar('/precision',precision)
        tf.summary.scalar('/recall',recall)
        tf.summary.scalar('/jaccard',jaccard)
        #tf.scalar_summary('/regularization_loss', regularization_loss)

        merged_summary = tf.summary.merge_all()

        ########################
        
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

            curr_seq = np.random.choice(all_seqs)

            input_provider = inp_provider_dict[curr_seq]
            reinit_branch(sess, curr_seq)


            while global_step_var.eval() <= max_iters:

                # select input provider
                next_batch = input_provider.sequence_batch_itr(batch_size)


                for i, sequence_batch in enumerate(next_batch,1):
                    step = global_step_var.eval()
                    if step > max_iters:
                        break

                    result = sess.run([apply_gradient_op, loss,merged_summary,acc,precision,recall,jaccard,tp_tensor,
                                       fn_tensor,fp_tensor,confusion_matrix],
                                      feed_dict={inp_branch1:sequence_batch.images,
                                                label:sequence_batch.labels,
                                                weights:sequence_batch.weights,
                                                is_training_pl:True})
                    loss_value = result[1]
                    acc_value = result[3]
                    precision_value = result[4]
                    recall_value = result[5]
                    jaccard_value = result[6]

                    logger.info('seq:{} iters:{}, seq_no:{} loss :{} accuracy:{} p:{} r:{} j:{}'.format(curr_seq,step,
                                                   i, loss_value,acc_value,precision_value,recall_value,jaccard_value))
                    
                    if step%100 ==0:
                        #import pdb
                        #pdb.set_trace()
                        train_summary_writer.add_summary(result[2], step )

                    if step % 1000 == 0:
                        logger.info('Saving weights.')
                        saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                        logger.info('Flushing .')                        
                        train_summary_writer.flush()

                    if step % (4*MAX_ITER_PER_SEQ) == 0:
                        test_out_dir = os.path.join('test_out',RUN_ID,'iter-{}'.format(step))
                        ensure_dir(test_out_dir)
                        test_merge_osvos_net.test_network(sess,net_dict,test_out_dir,pfc)

                    if step % MAX_ITER_PER_SEQ == 0:
                        
                        curr_seq = np.random.choice(all_seqs)
                        logger.info('switching. input provider seq:{}'.format(curr_seq))
                        input_provider = inp_provider_dict[curr_seq]
                        reinit_branch(sess,curr_seq)
                        break
                
            train_summary_writer.close()


    
    
