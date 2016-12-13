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


    def __init__(self,inp, is_training = False , head = RESNET_50):
        '''
        Constructor
        '''
        self.inp = inp
        self.head = head

        self.is_training_pl = is_training
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
                
                tf.cond(self.is_training_pl, lambda: tf.nn.dropout(net, keep_prob=0.5), lambda: net)
                
                net=slim.fully_connected(net,(56*56),activation_fn = None,scope='linear2')
                
                end_points = dict(tf.get_collection(end_points_collection))
                
        return net,end_points
       
    def build(self, inp):
        # Create the network
        
        # Head
        if self.head  == VGG_16:
            basemodel_arg_scope = vgg.vgg_arg_scope()         
            with slim.arg_scope(basemodel_arg_scope):
                net,end_points_head = vgg.vgg_16(inp)
        elif self.head == RESNET_50:
            weight_decay = 0.00005

            with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
                net, end_points_head = resnet_v1.resnet_v1_50(inp,
                                                    is_training=self.is_training_pl,
                                                    global_pool=False,
                                                    output_stride=16)
        
        # Tail
        net,end_points_tail = self.__add_extra_layers(net)  
        
        end_points = end_points_head.copy()
        end_points.update(end_points_tail)
        self.net = net
        self.end_points = end_points
        self.end_points['prediction'] = tf.reshape(tf.sigmoid(self.net),[-1,56,56,1])
    
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
    
    def get_init_prev_mask_filters(self,temp_val):
              
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
        return filt_array
    
    def get_init_prev_rgb_filters(self,temp_val):
              
        num_filters = temp_val.shape[-1]
        kernal_size = temp_val.shape[0:2]  
        filt_list = []
        
        for i in range(num_filters):
            mean_rfilt = temp_val[:,:,0,i].mean()
            std_rfilt = temp_val[:,:,0,i].std()
            
            mean_gfilt = temp_val[:,:,1,i].mean()
            std_gfilt = temp_val[:,:,1,i].std()
            
            mean_bfilt = temp_val[:,:,2,i].mean()
            std_bfilt = temp_val[:,:,2,i].std()
            
            rfilt = np.random.normal(mean_rfilt,std_rfilt,kernal_size)
            gfilt = np.random.normal(mean_gfilt,std_gfilt,kernal_size)
            bfilt = np.random.normal(mean_bfilt,std_bfilt,kernal_size)
            rgb_filt = np.dstack((rfilt,gfilt,bfilt))
            rgb_filt = rgb_filt[:,:,:,np.newaxis]
            filt_list.append(rgb_filt)   
                    
            
        filt_array = np.concatenate(filt_list,axis=3) 
        return filt_array        
    
    def im_predict(self,session,batch):
        if len(batch.shape) != 4:
            raise ValueError('Input must be of size [batch_size, height, width, C>0]')
        
        predict = self.end_points['prediction']
        result = session.run([predict],feed_dict={self.inp:batch,
                                                  self.is_training_pl:False})
        
        return result[0][0,:,:,0]
    
    def add_filter_channel_and_init(self,session,head,checkpoint_file,filter_name,previous_filter_size):
        
        inp_channels = self.inp.get_shape()[3]
        if (not (inp_channels == 4 or inp_channels == 7 )):
            raise ValueError('Inappropriate size of input placeholder. Supported num channels are 4 or 7')
        
        
        # Create a temp variable to restore the previous value to
        tempVar = tf.get_variable("temp/w",previous_filter_size)
        
        # Restore the previous value to the temp variable
        varMap = {head+'/'+filter_name:tempVar} 
        restorer1 = tf.train.Saver(varMap)  
        restorer1.restore(session,checkpoint_file)
        temp_val = tempVar.eval()
         
    
        # Add extra channel for previous mask channel               
        prev_mask_filt_array = self.get_init_prev_mask_filters(temp_val)                
        temp_mod_val = np.concatenate((temp_val,prev_mask_filt_array),axis=2) 
        
        if (inp_channels == 7):
            # Add extra channel for previous mask channel               
            prev_rgb_array = self.get_init_prev_rgb_filters(temp_val)                
            temp_mod_val = np.concatenate((temp_mod_val,prev_rgb_array),axis=2) 
             
        # Restore it to the actual network
        with tf.variable_scope(head,reuse=True):    
            w=tf.get_variable(filter_name)
            assign_op = w.assign(temp_mod_val)
            session.run(assign_op)   