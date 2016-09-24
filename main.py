# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:27:22 2016

@author: george
"""

from models import vgg
import tensorflow as tf
import numpy as np
from dataprovider.SampleInputProvider import SampleInputProvider
from utils.logger import getLogger
from skimage import io
import time

slim = tf.contrib.slim

HEAD = 'vgg_16'
TAIL = 'tail'

LOG_DIR = 'log/'+time.strftime("%Y%m%d-%H%M%S")


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

def initHead(session,net,checkpointFile):
    
    firstConvLayerWeightName = 'conv1/conv1_1/weights'
    tempVar = tf.get_variable("temp/w",[3,3,3,64])
    
    varMap = {HEAD+'/'+firstConvLayerWeightName:tempVar}
    restorer1 = tf.train.Saver(varMap)  
    restorer1.restore(session,checkpointFile)
    temp_val = tempVar.eval()
    
    num_filters = temp_val.shape[-1]
    num_old_channels = temp_val.shape[2]
    kernal_size = temp_val
    
    for i in range(num_filters):
        mean_filt = temp_val[:,:,:,i].mean()
        std_filt = temp_val[:,:,:,i].std()
        
        filt = np.random.normal(mean_filt,std_filt,kernal_size)
        filt = filt[:,:,np.newaxis,np.newaxis]
        
        
        
        
    
    tempModVal = np.insert(temp_val,3,0,axis=2)    
    with tf.variable_scope( HEAD,reuse=True):    
        w=tf.get_variable(firstConvLayerWeightName)
        assign_op = w.assign(tempModVal)
        session.run(assign_op)
    

    varToRestore = slim.get_variables_to_restore(include = [HEAD],
                                        exclude=[HEAD+'/'+firstConvLayerWeightName])
    restorer2 = tf.train.Saver(varToRestore)
    restorer2.restore(session,checkpointFile)
    
    

logger = getLogger()
with tf.Graph().as_default():
    basemodel_arg_scope = vgg.vgg_arg_scope()
          
    with slim.arg_scope(basemodel_arg_scope):
        # Create placeholders for input and output
        inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
        label =  tf.placeholder(tf.float32,shape=[None,224,224],name='label')
       
        #Resize label. Hack add third dim 
        h_label = tf.expand_dims(label,2)        
        label_resized = tf.image.resize_images(h_label,56,56)
        label_resized = tf.squeeze(label_resized)
        
        label_reshaped = tf.reshape(label_resized,[-1,(56*56)]) 
        
        # Create the network
        net,endpoints = vgg.vgg_16(inp)
        net = add_extra_layers(net)
        #net = tf.reshape(net,[-1,56,56])  
        
        # Add the loss layer
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                net, label_reshaped))
        
        # Declare the optimizer
        global_step = tf.Variable(0, trainable=False)
        
        learning_rate = tf.train.exponential_decay(0.001, global_step, 10,
                                   0.1, staircase=True)
        
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        gradients = optimizer.compute_gradients(loss)
        
        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
                
        
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        
        max_epochs = 100
        batch_size = 1
        
        # Input Provider
        inputProvider = SampleInputProvider()    
        init_op = tf.initialize_variables([global_step])
        
        # Testing
        out=tf.reshape(tf.sigmoid(net),[-1,56,56,1])
        
        tf.image_summary('/output',out)
        tf.scalar_summary('/loss', loss)
    
    with tf.Session() as sess:
        sess.run(init_op)
        # Initialize the weights
        initHead(sess,net,'checkpoints/vgg_16.ckpt')
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

            if step % 10 == 0:
                
                io.imshow(out.eval())
                #pass
                #logger.info('Saving weights.')
                #saver.save(session, LEARNED_WEIGHTS_FILENAME)
            
        logger.info('epoc:{}, loss:{}'.format(step, loss_value))  
        
        # Display
      

        
    
print('End')


