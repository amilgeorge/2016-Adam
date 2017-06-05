# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:27:22 2016

@author: george
"""

from models import vgg
from models import resnet_v1
import tensorflow as tf
import numpy as np
from dataprovider.sampleinputprovider import SampleInputProvider
from common.logger import getLogger
from common.diskutils import ensure_dir
from skimage import io
from net.coarsenet import CoarseNet
import time
import os
from ops.loss import weighted_cross_entropy
import pickle

slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1_50'
HEAD = RESNET_50
TAIL = 'tail'

RUN_ID = "c-ch7-aug4-weightedlossP3-3"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)


    
if __name__ == '__main__':
    
    ensure_dir(EXP_DIR)
    ensure_dir(LOGS_DIR)        
    logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))         
    
    
    with tf.Graph().as_default():
    
        # Create placeholders for input and output
        inp = tf.placeholder(tf.float32,shape=[None,224,224,7],name='input')
        label =  tf.placeholder(tf.float32,shape=[None,224,224],name='label')
        weights = tf.placeholder(tf.float32,shape=[None,56,56],name='weights')
        is_training_pl = tf.placeholder(tf.bool,name="coarsenet_is_training")
   
        #Resize label. Hack add third dim 
        h_label = tf.expand_dims(label,3)        
        label_resized = tf.image.resize_images(h_label, size=[56, 56])
        #label_resized = tf.squeeze(label_resized)
            
        label_reshaped = tf.reshape(label_resized,[-1,(56*56)])

        weights_reshaped = tf.reshape(weights, [-1,(56*56)])
        
        coarse_net = CoarseNet(inp,head=HEAD,is_training=is_training_pl)
        net,end_points = coarse_net.net,coarse_net.end_points
            
        # Add the loss layer
        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        loss = weighted_cross_entropy( net, label_reshaped,weights_reshaped)
        loss = tf.reduce_mean(loss) + regularization_loss

        # Unweigheted loss
        loss_unweighted = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=label_reshaped))
            
        # Declare the optimizer
        global_step_var = tf.Variable(1, trainable=False)
            
        learning_rate = tf.train.exponential_decay(0.01, global_step_var, 10,
                                       0.1, staircase=True)
            
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        #optimizer = tf.train.MomentumOptimizer(0.01,0.9)
            
        gradients = optimizer.compute_gradients(loss)
                               
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step_var)
            
        max_iters = 30000
        batch_size = 16
            
        # Input Provider
        inputProvider = SampleInputProvider(is_coarse=True,is_dummy=False)    
        init_op = tf.variables_initializer([global_step_var])
            


        ##### Summaries #######
        out=tf.reshape(tf.sigmoid(net),[-1,56,56,1])
        predictions = tf.round(out)
        prediction_reshaped = tf.cast(tf.reshape(predictions, [-1]),tf.int64)
        label_reshaped = tf.cast(tf.reshape(label_reshaped, [-1]),tf.int64)


        acc = tf.contrib.metrics.accuracy(prediction_reshaped, label_reshaped)
        confusion_matrix = tf.contrib.metrics.confusion_matrix(prediction_reshaped, label_reshaped)

        tp_tensor = confusion_matrix[1, 1]
        fp_tensor = confusion_matrix[1, 0]
        fn_tensor = confusion_matrix[0, 1]

        precision = tp_tensor / (tp_tensor + fp_tensor)
        recall = tp_tensor / (tp_tensor + fn_tensor)

        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        tf.summary.scalar('/loss', loss)
        tf.summary.scalar('/accuracy', acc)
        tf.summary.scalar('/precision', precision)
        tf.summary.scalar('/recall', recall)
        tf.summary.scalar('/regularization_loss', regularization_loss)
        #tf.scalar_summary('/loss_unweighted',loss_unweighted)
        merged_summary = tf.summary.merge_all()

        VALIDATION_SUMMARIES = 'validation_summaries'

        val_loss_pl = tf.placeholder(tf.float32)
        val_acc_pl = tf.placeholder(tf.float32)
        val_precision_pl = tf.placeholder(tf.float32)
        val_recall_pl = tf.placeholder(tf.float32)
        #sval_loss_unweighted_pl = tf.placeholder(tf.float32)

        #sval_loss_unweighted_summary = tf.scalar_summary('/val/loss_unweighted', val_loss_unweighted_pl,                                                      collections=VALIDATION_SUMMARIES)
        val_loss_summary = tf.summary.scalar('/val/loss', val_loss_pl, collections=VALIDATION_SUMMARIES)
        val_acc_summary = tf.summary.scalar('/val/accuracy', val_acc_pl, collections=VALIDATION_SUMMARIES)
        val_precision_summary = tf.summary.scalar('/val/precision', val_precision_pl, collections=VALIDATION_SUMMARIES)
        val_recall_summary = tf.summary.scalar('/val/recall', val_recall_pl, collections=VALIDATION_SUMMARIES)

        merged_val_summary = tf.summary.merge(
            [val_loss_summary, val_acc_summary, val_precision_summary, val_recall_summary],
            collections=None)
        ########################


        saver = tf.train.Saver(max_to_keep = 30)
        #saver.export_meta_graph("metagraph.meta", as_text=True)    
        def perform_validation(session, step, summary_writer):

            losses = []
            accuracies = []
            precisions = []
            recalls = []
            val_data = inputProvider.val_seq_batch_itr(batch_size)
            for i, sequence_batch in enumerate(val_data):
                result = session.run([loss, acc, precision, recall, confusion_matrix],
                                     feed_dict={inp: sequence_batch.images,
                                                label: sequence_batch.labels,
                                                weights: sequence_batch.weights,
                                                is_training_pl: False})
                loss_value = result[0]
                acc_value = result[1]
                precision_value = result[2]
                recall_value = result[3]
                losses.append(loss_value)
                accuracies.append(acc_value)
                precisions.append(precision_value)
                recalls.append(recall_value)
                logger.info('val iters:{}, seq_no:{} loss :{} acc:{}'
                            'p:{} r:{}  '.format(step, i, loss_value, acc_value, precision_value, recall_value))

            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)

            feed = {val_loss_pl: avg_loss,
                    val_acc_pl: avg_accuracy,
                    val_precision_pl: avg_precision,
                    val_recall_pl: avg_recall
                    }

            val_summary = session.run([merged_val_summary], feed_dict=feed)
            summary_writer.add_summary(val_summary[0], step)
                
  
                
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
           
            
            checkpoint_file = tf.train.latest_checkpoint(EXP_DIR);
            if checkpoint_file:
                restorer = tf.train.Saver([global_step_var])
                coarse_net.initialize(sess,checkpoint_file)
                restorer.restore(sess, checkpoint_file)

            else :   
                logger.info('Initializing using default weights')
                # Initialize the weights
                coarse_net.initialize(sess)
                sess.run(init_op)

            train_summary_writer = tf.summary.FileWriter(EVENTS_DIR + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(EVENTS_DIR + '/test')
            while global_step_var.eval() <= max_iters:
                #logger.info('Executing step:{}'.format(step))
                next_batch = inputProvider.sequence_batch_itr(batch_size)
                for i, sequence_batch in enumerate(next_batch,1):
                    step = global_step_var.eval()
                    if (step > max_iters):
                        break

                    result = sess.run([apply_gradient_op, loss, merged_summary, acc, precision, recall, tp_tensor,
                                       fn_tensor, fp_tensor, confusion_matrix],
                                      feed_dict={inp: sequence_batch.images,
                                                 label: sequence_batch.labels,
                                                 weights: sequence_batch.weights,
                                                 is_training_pl: True})
                    loss_value = result[1]
                    acc_value = result[3]
                    precision_value = result[4]
                    recall_value = result[5]
                    cm=result[9]
                    logger.info(cm)
                    logger.info('iters:{}, seq_no:{} loss :{} accuracy:{} p:{} r:{}'.format(step, i, loss_value,acc_value,precision_value,recall_value))

                    
                    if step%100 ==0:
                        train_summary_writer.add_summary(result[2], step )
    
                    
                    #io.imshow(out.eval())
                    #pass
                    if step % 200 == 0:
                        perform_validation(sess,step,test_summary_writer)

                    if step % 500 ==0:
                        logger.info('Saving weights.')
                        saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                        logger.info('Flushing .')
                        train_summary_writer.flush()

            train_summary_writer.close()
            test_summary_writer.close()
    
    
    
