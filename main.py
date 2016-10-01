# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:27:22 2016

@author: george
"""

from models import vgg
from models import resnet_v1
import tensorflow as tf
import numpy as np
from dataprovider import SampleInputProvider
from common.logger import getLogger
from common.diskutils import ensure_dir
from skimage import io
import time
import os

slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1'
HEAD = RESNET_50
TAIL = 'tail'

RUN_ID = "test3"
LOG_DIR = os.path.join('log',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)


def add_extra_layers(net):
    with tf.variable_scope(TAIL) as sc:
        end_points_collection = sc.name + '_end_points'
        net=slim.conv2d(net,512,[1,1],scope = 'conv1')
        net=slim.flatten(net,scope='flatten1')
        net=slim.fully_connected(net,512,activation_fn=None,scope='linear1')
        net=slim.fully_connected(net,(56*56),activation_fn = None,scope='linear2')
    
    return net
    
def getVariablesToRestore(net,excluded):
    
    allvars = []
   
    return [v for v in allvars if v not in excluded]
    
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
    add_filter_channel_and_init(session, head, first_layer_filter_name, previous_filter_size)
    
    # initialize remaining network
    
    varToRestore = slim.get_variables_to_restore(include = [head],
                                        exclude=[head+'/'+first_layer_filter_name])
    restorer2 = tf.train.Saver(varToRestore)
    restorer2.restore(session,checkpointFile)
        
def add_filter_channel_and_init(session,head,filter_name,previous_filter_size):
    
    # Create a temp variable to restore the previous value to
    tempVar = tf.get_variable("temp/w",previous_filter_size)
    
    # Restore the previous value to the temp variable
    varMap = {head+'/'+filter_name:tempVar} 
    restorer1 = tf.train.Saver(varMap)  
    restorer1.restore(session,checkpointFile)
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
    
def initVGG16Head(session,net,checkpointFile):
    
    firstConvLayerWeightName = 'conv1/conv1_1/weights'
    tempVar = tf.get_variable("temp/w",[3,3,3,64])
    
    varMap = {HEAD+'/'+firstConvLayerWeightName:tempVar}
    restorer1 = tf.train.Saver(varMap)  
    restorer1.restore(session,checkpointFile)
    temp_val = tempVar.eval()
    
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
        
    
    #temp_mod_val = np.insert(temp_val,3,0,axis=2)    

    

    varToRestore = slim.get_variables_to_restore(include = [HEAD],
                                        exclude=[HEAD+'/'+firstConvLayerWeightName])
    restorer2 = tf.train.Saver(varToRestore)
    restorer2.restore(session,checkpointFile)
    
def build_model(input, head):
    # Create the network
    
    # Head
    if head  == VGG_16:
        basemodel_arg_scope = vgg.vgg_arg_scope()         
        with slim.arg_scope(basemodel_arg_scope):
            net,endpoints = vgg.vgg_16(input)
    elif head == RESNET_50:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(input,
                                                is_training=True,
                                                global_pool=False,
                                                output_stride=16)
    
    # Tail
    net = add_extra_layers(net)  
    
    return net  


if __name__ == '__main__':
            
    logger = getLogger()         
    
    ensure_dir(EXP_DIR)
    
    with tf.Graph().as_default():
    
        # Create placeholders for input and output
        inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
        label =  tf.placeholder(tf.float32,shape=[None,224,224],name='label')
           
        #Resize label. Hack add third dim 
        h_label = tf.expand_dims(label,3)        
        label_resized = tf.image.resize_images(h_label,56,56)
        #label_resized = tf.squeeze(label_resized)
            
        label_reshaped = tf.reshape(label_resized,[-1,(56*56)]) 
            
            
        net = build_model(inp, HEAD)
            
        # Add the loss layer
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    net, label_reshaped))
            
        # Declare the optimizer
        global_step = tf.Variable(0, trainable=False)
            
        learning_rate = tf.train.exponential_decay(0.01, global_step, 10,
                                       0.1, staircase=True)
            
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        #optimizer = tf.train.MomentumOptimizer(0.01,0.9)
            
        gradients = optimizer.compute_gradients(loss)
            
        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
                    
            
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
            
        max_epochs = 150
        batch_size = 16
            
        # Input Provider
        inputProvider = SampleInputProvider()    
        init_op = tf.initialize_variables([global_step])
            
        # Testing
        out=tf.reshape(tf.sigmoid(net),[-1,56,56,1])
            
        tf.image_summary('/label',label_resized)
        tf.image_summary('/output',out)
        tf.scalar_summary('/loss', loss)
        
        saver = tf.train.Saver(max_to_keep = 10)
        #saver.export_meta_graph("metagraph.meta", as_text=True)        
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init_op)
            
            checkpoint_file = tf.train.latest_checkpoint(EXP_DIR);
            if checkpoint_file:
                logger.info('Initializing using checkpoint file:{}'.format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            else :   
                logger.info('Initializing using default weights')
                # Initialize the weights
                initHead(sess,HEAD,net)
                initTail(sess,net)
            
            merged_summary = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph)
            
            for step in range(max_epochs):
                logger.info('Executing step:{}'.format(step))
                next_batch = inputProvider.sequence_batch_itr(batch_size)
                for i, sequence_batch in enumerate(next_batch):
                    #logger.debug('epoc:{}, seq_no{}'.format(step, i))
                    result = sess.run([apply_gradient_op, loss, merged_summary], 
                                      feed_dict={inp:sequence_batch.images,
                                                label:sequence_batch.labels})
                    loss_value = result[1]
                    logger.info('epoc:{}, seq_no:{} loss :{}'.format(step, i, loss_value))
                    summary_writer.add_summary(result[2], step * i + i)
    
                    
                    #io.imshow(out.eval())
                    #pass
                logger.info('Saving weights.')
                saver.save(sess, os.path.join(EXP_DIR,'epoch'),global_step = step)
                logger.info('Flushing .')
                summary_writer.flush()
                
            summary_writer.close()
    
    
    
