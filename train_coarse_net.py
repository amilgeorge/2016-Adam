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

RUN_ID = "coarse-weighted-f5-5"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)



if __name__ == '__main__':
    
    ensure_dir(EXP_DIR)
    ensure_dir(LOGS_DIR)        
    logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))         
    
    
    with tf.Graph().as_default():
    
        # Create placeholders for input and output
        inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
        label =  tf.placeholder(tf.float32,shape=[None,224,224],name='label')
        weights = tf.placeholder(tf.float32,shape=[None,56,56],name='weights')
   
        #Resize label. Hack add third dim 
        h_label = tf.expand_dims(label,3)        
        label_resized = tf.image.resize_images(h_label,56,56)
        #label_resized = tf.squeeze(label_resized)
            
        label_reshaped = tf.reshape(label_resized,[-1,(56*56)]) 
        weights_reshaped = tf.reshape(weights, [-1,(56*56)])
        
        coarse_net = CoarseNet(inp,HEAD)
        net,end_points = coarse_net.net,coarse_net.end_points
            
        # Add the loss layer
        loss = weighted_cross_entropy( net, label_reshaped,weights_reshaped)
        loss = tf.reduce_mean(loss)

        # Unweigheted loss
        loss_unweighted = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(net, label_reshaped))
            
        # Declare the optimizer
        global_step_var = tf.Variable(0, trainable=False)
            
        learning_rate = tf.train.exponential_decay(0.01, global_step_var, 10,
                                       0.1, staircase=True)
            
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        #optimizer = tf.train.MomentumOptimizer(0.01,0.9)
            
        gradients = optimizer.compute_gradients(loss)
            
        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
                    
            
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step_var)
            
        max_iters = 150000
        batch_size = 16
            
        # Input Provider
        inputProvider = SampleInputProvider(is_coarse=True,is_dummy=True)    
        init_op = tf.initialize_variables([global_step_var])
            
        # Testing
        out=tf.reshape(tf.sigmoid(net),[-1,56,56,1])
            
        tf.image_summary('/label',label_resized)
        tf.image_summary('/output',out)
        tf.scalar_summary('/loss', loss)
        tf.scalar_summary('/loss_unweighted',loss_unweighted)
        
        saver = tf.train.Saver(max_to_keep = 10)
        #saver.export_meta_graph("metagraph.meta", as_text=True)    
  
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
            
            merged_summary = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(EVENTS_DIR, sess.graph)
            while global_step_var.eval() < max_iters:                              
                #logger.info('Executing step:{}'.format(step))
                next_batch = inputProvider.sequence_batch_itr(batch_size)
                for i, sequence_batch in enumerate(next_batch):
                    step = global_step_var.eval()
                    if (step >= max_iters):
                        break
                    result = sess.run([apply_gradient_op, loss, merged_summary,loss_unweighted], 
                                      feed_dict={inp:sequence_batch.images,
                                                label:sequence_batch.labels,
                                                weights:sequence_batch.weights})
                    loss_value = result[1]
                    loss_unweighted_value = result[3]
                    logger.info('iters:{}, seq_no:{} loss :{} loss_unweighted:{}'.format(step, i, loss_value,loss_unweighted_value))
                    
                    if step%100 ==0:
                        summary_writer.add_summary(result[2], step )
    
                    
                    #io.imshow(out.eval())
                    #pass
                    if step % 1000 ==0:
                        logger.info('Saving weights.')
                        saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                        logger.info('Flushing .')
                        summary_writer.flush()
                
            summary_writer.close()
    
    
    
