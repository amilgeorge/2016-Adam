'''
Created on Dec 1, 2016

@author: george
'''
# -*- coding: utf-8 -*-
from net.segnet2 import NUM_CLASSES
import tensorflow as tf
import numpy as np
from dataprovider.mergenetdataprovider import InputProvider

from common.logger import getLogger
from common.diskutils import ensure_dir
from net import segnet2 as segnet
import time
import os
import test_merge_net
#import tracemalloc
#import gc

slim = tf.contrib.slim

BRANCH_1_OFFSET = 1
BRANCH_2_OFFSET = 0
CHECKPOINT_BRANCH1 = 'exp/segnet480pvgg-wl-dp2-osvos-val-O1-3/iters-45000'
CHECKPOINT_BRANCH2 = 'exp/segnet480pvgg-wl-dp2-osvos-val-O0-4/iters-45000'


RUN_ID = "mergenetvgg-B1O{}-B2O{}-adamlr000134".format(BRANCH_1_OFFSET,BRANCH_2_OFFSET)


EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

IMG_HEIGHT = 480
IMG_WIDTH = 854

    
if __name__ == '__main__':

    #tracemalloc.start()

    #gc.enable()
    #gc.set_debug(gc.DEBUG_LEAK)

    ensure_dir(EXP_DIR)
    ensure_dir(LOGS_DIR)        
    logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))

    RNG_SEED = 3
    np.random.seed(RNG_SEED)

    with tf.Graph().as_default():
        tf.set_random_seed(1)

        NUM_CLASSES = segnet.NUM_CLASSES
        # Create placeholders for input and output
        inp_branch1 = tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,7],name='input_branch1')
        inp_branch2 = tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,7],name='input_branch2')

        label =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH],name='label')
        weights =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH],name='weights')

        is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")



        loss,logit = segnet.inference_merge_two_branch(inp_branch1,inp_branch2, label, is_training_pl,weights)

        loss_averages_op = segnet._add_loss_summaries(loss)
        #segnet.prepare_encoder_parameters()


        logit = tf.reshape(logit, (-1, NUM_CLASSES))
        out=tf.reshape(tf.nn.softmax(logit),[-1,IMG_HEIGHT,IMG_WIDTH,2])

        net_dict = {"inp_branch1_pl": inp_branch1,
                    "inp_branch2_pl": inp_branch2,
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
            optimizer = tf.train.AdamOptimizer(0.0001)
            #optimizer = tf.train.MomentumOptimizer(0.001,0.9)

            gradients = optimizer.compute_gradients(loss)
            
        
                    
            
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step_var)
        #apply_gradient_op = segnet.train(loss, global_step_var)    
        max_iters = 45000
        batch_size = segnet.BATCH_SIZE
            
        # Input Provider
        #inputProvider = SampleInputProvider(resize=[IMG_HEIGHT,IMG_WIDTH],is_dummy=False)
        inputProvider = InputProvider(BRANCH_1_OFFSET,BRANCH_2_OFFSET)


    
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
        tf.image_summary('/output',tf.expand_dims(out_reshaped[:,:,:,1],3))
        tf.image_summary('/label',tf.expand_dims(label,3))

        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
                
        tf.scalar_summary('/loss', loss)
        tf.scalar_summary('/accuracy',acc)
        tf.scalar_summary('/precision',precision)
        tf.scalar_summary('/recall',recall)
        tf.scalar_summary('/jaccard',jaccard)
        #tf.scalar_summary('/regularization_loss', regularization_loss)

        merged_summary = tf.merge_all_summaries()
        
        VALIDATION_SUMMARIES = 'validation_summaries'
        
        val_loss_pl = tf.placeholder(tf.float32)
        val_acc_pl = tf.placeholder(tf.float32)
        val_precision_pl = tf.placeholder(tf.float32)
        val_recall_pl = tf.placeholder(tf.float32)
        val_jaccard_pl = tf.placeholder(tf.float32)


        val_loss_summary = tf.scalar_summary('/val/loss', val_loss_pl,collections=VALIDATION_SUMMARIES)
        val_acc_summary = tf.scalar_summary('/val/accuracy', val_acc_pl,collections=VALIDATION_SUMMARIES)
        val_precision_summary = tf.scalar_summary('/val/precision', val_precision_pl,collections=VALIDATION_SUMMARIES)
        val_recall_summary = tf.scalar_summary('/val/recall', val_recall_pl ,collections=VALIDATION_SUMMARIES)
        val_jaccard_summary = tf.scalar_summary('/val/jaccard', val_jaccard_pl ,collections=VALIDATION_SUMMARIES)


        merged_val_summary = tf.merge_summary([val_loss_summary,val_acc_summary,val_precision_summary,val_recall_summary,val_jaccard_summary ],
                                              collections=None)
        ########################
        
        saver = tf.train.Saver(max_to_keep = 10)
        #saver.export_meta_graph("metagraph.meta", as_text=True)    
        def perform_validation(session,step,summary_writer):

            losses = []
            accuracies = []
            precisions = []
            recalls = []
            jaccards = []
            val_data = inputProvider.val_seq_batch_itr(batch_size)
            for i, sequence_batch in enumerate(val_data):
                result = session.run([loss,acc,precision,recall,jaccard,confusion_matrix],
                                        feed_dict={inp:sequence_batch.images,
                                        label:sequence_batch.labels,
                                        weights:sequence_batch.weights,
                                        is_training_pl:False})#,
                                        #keep_prob :1.0})
                loss_value = result[0]
                acc_value = result[1]
                precision_value = result[2]
                recall_value = result [3]
                jaccard_value = result[4]
                losses.append(loss_value)
                accuracies.append(acc_value)
                precisions.append(precision_value)
                recalls.append(recall_value)
                jaccards.append(jaccard_value)
                logger.info('val iters:{}, seq_no:{} loss :{} acc:{}'
                                'p:{} r:{} j:{} '.format(step, i, loss_value, acc_value,precision_value,recall_value,jaccard_value))

            
            avg_loss = sum(losses)/len(losses)
            avg_accuracy  = sum(accuracies)/len(accuracies)
            avg_precision = sum(precisions)/len(precisions)
            avg_recall = sum(recalls)/len(recalls)
            avg_jaccard = sum(jaccards)/len(jaccards)


            feed = {val_loss_pl: avg_loss,
                    val_acc_pl:avg_accuracy,
                    val_precision_pl:avg_precision,
                    val_recall_pl:avg_recall,
                    val_jaccard_pl:avg_jaccard
                    }
            
            val_summary = session.run([merged_val_summary],feed_dict = feed)
            summary_writer.add_summary(val_summary[0],step)


        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

            checkpoint_file = tf.train.latest_checkpoint(EXP_DIR)

            if checkpoint_file:
                logger.info('using checkpoint :{}'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            else :   
                # Build an initialization operation to run below.
                init = tf.initialize_all_variables()

                # Start running operations on the Graph.
                sess.run(init)

                segnet.initialize_merge_net(sess,CHECKPOINT_BRANCH1,CHECKPOINT_BRANCH2)

            train_summary_writer = tf.train.SummaryWriter(EVENTS_DIR + '/train', sess.graph)
            test_summary_writer = tf.train.SummaryWriter(EVENTS_DIR + '/test')

            while global_step_var.eval() <= max_iters:
                #logger.info('Executing step:{}'.format(step))
                next_batch = inputProvider.sequence_batch_itr(batch_size)
                for i, sequence_batch in enumerate(next_batch,1):
                    step = global_step_var.eval()
                    if (step > max_iters):
                        break

                    result = sess.run([apply_gradient_op, loss,merged_summary,acc,precision,recall,jaccard,tp_tensor,
                                       fn_tensor,fp_tensor,confusion_matrix],
                                      feed_dict={inp_branch1:sequence_batch.images_branch1,
                                                 inp_branch2:sequence_batch.images_branch2,
                                                label:sequence_batch.labels,
                                                weights:sequence_batch.weights,
                                                is_training_pl:True})#,
                                                #keep_prob: 0.5})
                    loss_value = result[1]
                    acc_value = result[3]
                    precision_value = result[4]
                    recall_value = result[5]
                    jaccard_value = result[6]

                    logger.info('iters:{}, seq_no:{} loss :{} accuracy:{} p:{} r:{} j:{}'.format(step, i, loss_value,acc_value,precision_value,recall_value,jaccard_value))
                    
                    if step%100 ==0:
                        #import pdb
                        #pdb.set_trace()
                        train_summary_writer.add_summary(result[2], step )

                    
                    #io.imshow(out.eval())
                    #pass


                    if step % 1000 == 0:
                        logger.info('Saving weights.')
                        saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                        logger.info('Flushing .')                        
                        train_summary_writer.flush()
                        test_summary_writer.flush()

                    if step % 1000 == 0:
                        test_out_dir = os.path.join('test_out',RUN_ID,'iter-{}'.format(step))
                        ensure_dir(test_out_dir)
                        test_merge_net.test_network(sess,net_dict,test_out_dir,BRANCH_1_OFFSET,BRANCH_2_OFFSET)
                
            train_summary_writer.close()
            test_summary_writer.close()
    
    
    
