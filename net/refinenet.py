'''
Created on Oct 1, 2016

@author: george
'''

from net.coarsenet import CoarseNet
from main import RESNET_50
import tensorflow as tf
slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1_50'

class RefineNet(object):
    '''
    classdocs
    '''


    def __init__(self, coarse_net_head):
        '''
        Constructor
        '''
        self.inp_sz = 224
        self.coarse_net_head = RESNET_50
        self.km = 32
        self.ks = 32
        self.out_stride = 16
        self.fsz = self.inp_sz/self.out_stride
        
    
    def build(self,inp):
        coarse_net_object = CoarseNet(self.coarse_net_head)
        net,end_points = coarse_net_object.build(inp)
        
        net,end_points_refine= self.__add_sharp_mask_modules(net, end_points)
        end_points.update(end_points_refine)
        
        return net,end_points
    
    def __add_horizontal_units(self,i,inp_h):
        with tf.variable_scope('horz'):
            num_inps = self.km/2^i
            if i == 0:  
                nhu1,nhu2,crop=1024,64,0
            elif i == 1:  
                nhu1,nhu2,crop = 512,64,-2
            elif i == 2:  
                nhu1,nhu2,crop = 256,64,-4
            elif i == 3:  
                nhu1,nhu2,crop = 64,32,-8
            
            net = tf.pad(inp_h, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,nhu2 , [3, 3], stride=1,
                               activation_fn=tf.nn.relu, scope='conv1')
            
            net = tf.pad(inp_h, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps , [3, 3], stride=1,
                               activation_fn=tf.nn.relu, scope='conv2')
            
            net = tf.pad(inp_h, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps/2 , [3, 3], stride=1,
                               activation_fn=None, scope='conv3')
        
        return net
        
    def __add_vertical_units(self,i,inp_v):
        with tf.variable_scope('vert'):
            num_inps = self.km/2^i
            net = tf.pad(inp_v, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps , [3, 3], stride=1,
                               activation_fn=tf.nn.relu, scope='conv1')
            net = tf.pad(inp_v, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps/2 , [3, 3], stride=1,
                               activation_fn=None, scope='conv2')

        return net
    
    def __add_refinement_module(self,i,inp_h,inp_v):
        with tf.variable_scope('refinement_%d' % (i + 1)):
            hnet = self.add_horizontal_units(i,inp_h)
            vnet = self.add_vertical_units(i,inp_v)
            
            # Upsample 
            net = tf.add(hnet,vnet)
            net = tf.nn.relu(net)
            shape = net.get_shape().as_list()
            target_shape = [shape[0],shape[1]*2,shape[2]*2,shape[3]]
            net = tf.image.resize_nearest_neighbor(net, target_shape, name = "Upsample")
        return net
    
    def __top_512d_tensor(self,end_points):
        """
        Return top layer(512d) tensor 
        """
        return end_points['tail/linear1']
        
    
    def __horizontal_input_tensors(self,end_points):
        # Keys to endpoints of RESNET 50 
        key14d = 'tail/conv1'
        key28d = 'resnet_v1_50/block1'
        key56d = 'resnet_v1_50/block1/unit_1/bottleneck_v1'
        key112d = 'resnet_v1_50/conv1'
        
        # Add them to a list
        hlist = [end_points[key14d],end_points[key28d],end_points[key56d],end_points[key112d]]
        return hlist
    def __add_sharp_mask_modules(self,net,end_points):

        with tf.variable_scope('Refine') as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.flatten],
                            outputs_collections=end_points_collection):
                # Horizontal output layers from the main net
                inp_h_list = self.__horizontal_input_tensors(end_points)
                
                #Last output tensor ... 512-d 
                top_layer = self.__top_512d_tensor(end_points)
                
                net = slim.fully_connected(top_layer,self.fsz*self.fsz*self.km,activation_fn = None,scope='sharp_linear')
                net = tf.reshape(net,[-1,self.fsz,self.fsz,self.km],name='sharp_reshape')
                for i in range(4):
                    net =  self.add_refinement_module(i,inp_h_list[i], net) 
                
                net = tf.pad(net, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
                net = slim.conv2d(net,1 , [3, 3], stride=1,
                               activation_fn=None, scope='conv')    
            end_points = dict(tf.get_collection(end_points_collection))
        return net,end_points
            
            
if __name__ == '__main__':
    inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
    builder = RefineNet(RESNET_50)
    refine_net,end_points = builder.build(inp)       