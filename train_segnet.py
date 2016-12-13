'''
Created on Dec 1, 2016

@author: george
'''
# -*- coding: utf-8 -*-
from net.segnet2 import NUM_CLASSES
from dask.compatibility import apply
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
from net import segnet2 as segnet
import time
import os
from ops.loss import weighted_cross_entropy
import pickle

slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1_50'
HEAD = RESNET_50
TAIL = 'tail'

RUN_ID = "segnet-ch7-9"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)


    
if __name__ == '__main__':
    
    ensure_dir(EXP_DIR)
    ensure_dir(LOGS_DIR)        
    logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))         
    
    
    with tf.Graph().as_default():
    
        NUM_CLASSES = segnet.NUM_CLASSES
        # Create placeholders for input and output
        inp = tf.placeholder(tf.float32,shape=[None,224,224,7],name='input')
        label =  tf.placeholder(tf.float32,shape=[None,224,224],name='label')
        is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")
   
     
                    
        loss,logit = segnet.inference(inp, label, is_training_pl)

        
        #out=tf.reshape(tf.nn.softmax(logit),[-1,224,224,2])
        logit = tf.reshape(logit, (-1, NUM_CLASSES))
        out = tf.reshape(tf.nn.softmax(logit),[-1,224,224,NUM_CLASSES])  
        # Declare the optimizer
        global_step_var = tf.Variable(1, trainable=False)
            
        learning_rate = tf.train.exponential_decay(0.01, global_step_var, 10,
                                       0.1, staircase=True)
            
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        #optimizer = tf.train.MomentumOptimizer(0.01,0.9)
            
        gradients = optimizer.compute_gradients(loss)
            
        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
                    
            
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step_var)
        #apply_gradient_op = segnet.train(loss, global_step_var)    
        max_iters = 10000
        batch_size = segnet.BATCH_SIZE
            
        # Input Provider
        inputProvider = SampleInputProvider(is_coarse=False,is_dummy=False)    
        #init_op = tf.initialize_variables([global_step_var])
            
    
            
        tf.image_summary('/output',tf.expand_dims(out[:,:,:,1],3))
        tf.image_summary('/label',tf.expand_dims(label,3))

        tf.scalar_summary('/loss', loss)

        
        merged_summary = tf.merge_all_summaries()

        
        VALIDATION_SUMMARIES = 'validation_summaries'
        
        val_loss_pl = tf.placeholder(tf.float32)
        val_loss_summary = tf.scalar_summary('/val/loss', val_loss_pl,collections=VALIDATION_SUMMARIES)
        
        merged_val_summary = tf.merge_summary([val_loss_summary],collections=None)
        
        saver = tf.train.Saver(max_to_keep = 50)
        #saver.export_meta_graph("metagraph.meta", as_text=True)    
        def perform_validation(session,step,summary_writer):

            losses = []
            val_data = inputProvider.val_seq_batch_itr(batch_size)
            for i, sequence_batch in enumerate(val_data):
                result = session.run([loss], 
                                    feed_dict={inp:sequence_batch.images,
                                    label:sequence_batch.labels,
                                    is_training_pl:False})
                loss_value = result[0]
                losses.append(loss_value)
                logger.info('val iters:{}, seq_no:{} loss :{} '.format(step, i, loss_value))

            
            avg_loss = sum(losses)/len(losses)

            feed = {val_loss_pl: avg_loss,
                    }
            
            val_summary = session.run([merged_val_summary],feed_dict = feed)
            summary_writer.add_summary(val_summary[0],step)
                
  
                
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
           
            
            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            # Start running operations on the Graph.
            sess.run(init)
            
            summary_writer = tf.train.SummaryWriter(EVENTS_DIR, sess.graph)
            while global_step_var.eval() < max_iters:                              
                #logger.info('Executing step:{}'.format(step))
                next_batch = inputProvider.sequence_batch_itr(batch_size)
                for i, sequence_batch in enumerate(next_batch,1):
                    step = global_step_var.eval()
                    if (step >= max_iters):
                        break
                    
                    result = sess.run([apply_gradient_op, loss,merged_summary], 
                                      feed_dict={inp:sequence_batch.images,
                                                label:sequence_batch.labels,
                                                is_training_pl:True})
                    loss_value = result[1]
                    logger.info('iters:{}, seq_no:{} loss :{}'.format(step, i, loss_value))
                    
                    if step%100 ==0:
                        summary_writer.add_summary(result[2], step )
    
                    
                    #io.imshow(out.eval())
                    #pass
                    if step % 200 == 0:
                        perform_validation(sess,step,summary_writer)

                    if step % 500 == 0:
                        logger.info('Saving weights.')
                        saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                        logger.info('Flushing .')                        
                        summary_writer.flush()
                
            summary_writer.close()
    
    
    
