'''
Created on Oct 1, 2016

@author: george
'''


from models import vgg
from models import resnet_v1
import tensorflow as tf
import numpy as np
from common import logger
from common.diskutils import ensure_dir
from skimage import io
import time
import os
import net


slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1_50'
TAIL = 'tail'

class CoarseNet(object):
    '''
    classdocs
    '''


    def __init__(self,inp, head,is_training = True):
        '''
        Constructor
        '''
        self.head = head
        self.is_training = is_training
        self.build(inp)
        #net,self.end_points = self.build(inp)
    
    def __add_extra_layers(self,net):
        with tf.variable_scope(TAIL) as sc:
            
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.flatten],
                        outputs_collections=end_points_collection):
                net=slim.conv2d(net,512,[1,1],scope = 'conv1')
                net=slim.flatten(net,scope='flatten1')
                net=slim.fully_connected(net,512,activation_fn=None,scope='linear1')
                net=slim.fully_connected(net,(56*56),activation_fn = None,scope='linear2')
                
                end_points = dict(tf.get_collection(end_points_collection))
                
        return net,end_points
       
    def build(self, inp):
        # Create the network
        
        # Head
        if self.head  == VGG_16:
            basemodel_arg_scope = vgg.vgg_arg_scope()         
            with slim.arg_scope(basemodel_arg_scope):
                net,endpoints_head = vgg.vgg_16(inp)
        elif self.head == RESNET_50:
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points_head = resnet_v1.resnet_v1_50(inp,
                                                    is_training=self.is_training,
                                                    global_pool=False,
                                                    output_stride=16)
        
        # Tail
        net,end_points_tail = self.__add_extra_layers(net)  
        
        end_points = end_points_head.copy()
        end_points.update(end_points_tail)
        self.net = net
        self.end_points = end_points
        #return net,end_points   
    
    def initialize(self,session,checkpoint_file = None):
        
        if checkpoint_file:
            vars_head = tf.get_collection(tf.GraphKeys.VARIABLES,scope = self.head)
            vars_tail = tf.get_collection(tf.GraphKeys.VARIABLES,scope = TAIL)
            vars_to_restore = vars_head + vars_tail
            restorer = tf.train.Saver(vars_to_restore)
            restorer.restore(session, checkpoint_file)
            
        else:
            self.__init_head(session)
            self.__init_tail(session)
        
    def __init_tail(self,session):
        varsTail = tf.get_collection(tf.GraphKeys.VARIABLES,scope = TAIL)
        init_op = tf.initialize_variables(varsTail)
        session.run(init_op)


    def __init_head(self,session):
        head = self.head
        if head == VGG_16:
            checkpoint_file = 'checkpoints/vgg_16.ckpt'
            first_layer_filter_name = 'conv1/conv1_1/weights'
            previous_filter_size = [3,3,3,64]
        elif head == RESNET_50:
            checkpoint_file = 'checkpoints/resnet_v1_50.ckpt'
            first_layer_filter_name = 'conv1/weights'
            previous_filter_size = [7,7,3,64]
        
        # initialize modified first filter    
        self.add_filter_channel_and_init(session, head,checkpoint_file, first_layer_filter_name, previous_filter_size)
        
        # initialize remaining network
        
        varToRestore = slim.get_variables_to_restore(include = [head],
                                            exclude=[head+'/'+first_layer_filter_name])
        restorer2 = tf.train.Saver(varToRestore)
        restorer2.restore(session,checkpoint_file)
            
    def add_filter_channel_and_init(self,session,head,checkpoint_file,filter_name,previous_filter_size):
        
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