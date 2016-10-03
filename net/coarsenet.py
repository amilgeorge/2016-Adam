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
import netconfig


slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1_50'
TAIL = 'tail'

class CoarseNet(object):
    '''
    classdocs
    '''


    def __init__(self, head):
        '''
        Constructor
        '''
        self.head = head
    
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
                                                    is_training=True,
                                                    global_pool=False,
                                                    output_stride=16)
        
        # Tail
        net,end_points_tail = self.__add_extra_layers(net)  
        
        end_points = end_points_head.copy()
        end_points.update(end_points_tail)
        return net,end_points      