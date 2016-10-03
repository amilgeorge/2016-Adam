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
import netconfig

slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1_50'
HEAD = RESNET_50
TAIL = 'tail'

RUN_ID = "test3"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)


    
    
def initTail(session,net):
    varsTail = tf.get_collection(tf.GraphKeys.VARIABLES,scope = TAIL)
    init_op = tf.initialize_variables(varsTail)
    session.run(init_op)


def initHead(session,head,net):
    if head == VGG_16:
        checkpoint_file = 'checkpoints/vgg_16.ckpt'
        first_layer_filter_name = 'conv1/conv1_1/weights'
        previous_filter_size = [3,3,3,64]
    elif head == RESNET_50:
        checkpoint_file = 'checkpoints/resnet_v1_50.ckpt'
        first_layer_filter_name = 'conv1/weights'
        previous_filter_size = [7,7,3,64]
    
    # initialize modified first filter    
    add_filter_channel_and_init(session, head,checkpoint_file, first_layer_filter_name, previous_filter_size)
    
    # initialize remaining network
    
    varToRestore = slim.get_variables_to_restore(include = [head],
                                        exclude=[head+'/'+first_layer_filter_name])
    restorer2 = tf.train.Saver(varToRestore)
    restorer2.restore(session,checkpoint_file)
        
def add_filter_channel_and_init(session,head,checkpoint_file,filter_name,previous_filter_size):
    
    # Create a temp variable to restore the previous value to
    tempVar = tf.get_variable("temp/w",previous_filter_size)
    
    # Restore the previous value to the temp variable
    varMap = {head+'/'+filter_name:tempVar} 
    restorer1 = tf.train.Saver(varMap)  
    restorer1.restore(session,checkpoint_file)
    temp_val = tempVar.eval()
    
    # Modify the filter shape to target shape
    num_filters = temp_val.shape[-1]
    kernal_size = temp_val.shape[0:2]    

    filt_list = []
    for i in range(num_filters):
        mean_filt = temp_val[:,:,:,i].mean()
        std_filt = temp_val[:,:,:,i].std()
        
        filt = np.random.normal(mean_filt,std_filt,kernal_size)
        filt = filt[:,:,np.newaxis,np.newaxis]
        filt_list.append(filt)   
                
        
    filt_array = np.concatenate(filt_list,axis=3)                
    temp_mod_val = np.concatenate((temp_val,filt_array),axis=2) 
    
    # Restore it to the actual network
    with tf.variable_scope(head,reuse=True):    
        w=tf.get_variable(filter_name)
        assign_op = w.assign(temp_mod_val)
        session.run(assign_op)
    



if __name__ == '__main__':
    
    ensure_dir(EXP_DIR)
    ensure_dir(LOGS_DIR)        
    logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))         
    
    
    with tf.Graph().as_default():
    
        # Create placeholders for input and output
        inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
        label =  tf.placeholder(tf.float32,shape=[None,224,224],name='label')
           
        #Resize label. Hack add third dim 
        h_label = tf.expand_dims(label,3)        
        label_resized = tf.image.resize_images(h_label,56,56)
        #label_resized = tf.squeeze(label_resized)
            
        label_reshaped = tf.reshape(label_resized,[-1,(56*56)]) 
            
        coarse_net_builder = CoarseNet(HEAD)
        net,end_points = coarse_net_builder.build(inp)
            
        # Add the loss layer
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    net, label_reshaped))
            
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
        inputProvider = SampleInputProvider()    
        init_op = tf.initialize_variables([global_step_var])
            
        # Testing
        out=tf.reshape(tf.sigmoid(net),[-1,56,56,1])
            
        tf.image_summary('/label',label_resized)
        tf.image_summary('/output',out)
        tf.scalar_summary('/loss', loss)
        
        saver = tf.train.Saver(max_to_keep = 10)
        #saver.export_meta_graph("metagraph.meta", as_text=True)        
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
           
            
            checkpoint_file = tf.train.latest_checkpoint(EXP_DIR);
            if checkpoint_file:
                logger.info('Initializing using checkpoint file:{}'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                #start_epoch = global_step_var.eval(sess)

            else :   
                logger.info('Initializing using default weights')
                # Initialize the weights
                initHead(sess,HEAD,net)
                initTail(sess,net)
                sess.run(init_op)
                #start_epoch = global_step_var.eval(sess)
            
            merged_summary = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(EVENTS_DIR, sess.graph)
            
            while global_step_var.eval() < max_iters:                              
                #logger.info('Executing step:{}'.format(step))
                next_batch = inputProvider.sequence_batch_itr(batch_size)
                for i, sequence_batch in enumerate(next_batch):
                    step = global_step_var.eval()
                    if (step >= max_iters):
                        break
                    #logger.debug('epoc:{}, seq_no{}'.format(step, i))
                    result = sess.run([apply_gradient_op, loss, merged_summary], 
                                      feed_dict={inp:sequence_batch.images,
                                                label:sequence_batch.labels})
                    loss_value = result[1]
                    logger.info('iters:{}, seq_no:{} loss :{}'.format(step, i, loss_value))
                    summary_writer.add_summary(result[2], step )
    
                    
                    #io.imshow(out.eval())
                    #pass
                logger.info('Saving weights.')
                saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                logger.info('Flushing .')
                summary_writer.flush()
                
            summary_writer.close()
    
    
    
