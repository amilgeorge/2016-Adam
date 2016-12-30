'''
Created on Oct 1, 2016

@author: george
'''

from net.coarsenet import CoarseNet
import tensorflow as tf
import net
from dataprovider.preprocess import vgg_preprocess
slim = tf.contrib.slim

VGG_16 = 'vgg_16'
RESNET_50 = 'resnet_v1_50'

REFINE_SCOPE = 'Refine'

class RefineNet(object):
    '''
    classdocs
    '''


    def __init__(self, inp, is_training=False,coarse_net_head=RESNET_50,coarse_checkpoint_file=None):
        '''
        Constructor
        '''
        self.inp = inp
        self.inp_sz = 224
        self.coarse_net_head = coarse_net_head
        self.coarse_checkpoint_file = coarse_checkpoint_file
        self.km = 32
        self.ks = 32
        self.out_stride = 16
        self.fsz = int(self.inp_sz/self.out_stride)
        self.is_training_pl = is_training
        self.stop_horizontal_grads = True
        self.build(inp)

    
    def build(self,inp):
        
        coarse_net_object = CoarseNet(inp,head=self.coarse_net_head,is_training=self.is_training_pl)
        self.coarse_net = coarse_net_object
        
        net = coarse_net_object.net
        end_points = coarse_net_object.end_points
        
        net,end_points_refine= self.__add_sharp_mask_modules(net, end_points)
        end_points.update(end_points_refine)
        
        self.net = net
        self.end_points = end_points
        self.end_points['prediction'] = tf.reshape(tf.sigmoid(self.net),[-1,224,224,1])
    
    def im_predict(self,session,batch):
        if len(batch.shape) != 4:
            raise ValueError('Input must be of size [batch_size, height, width, C>0]')
        
        predict = self.end_points['prediction']
        result = session.run([predict],feed_dict={self.inp:batch})
        
        return result[0][0,:,:,0]
        
    def refine_variables(self):
        """
        Get variables that belong to the refine network
        """
        return tf.get_collection(tf.GraphKeys.VARIABLES,scope = REFINE_SCOPE)

    def initialize(self,session,checkpoint_file = None):
        """
        Initialize the network
        
        """        
        ## Initialize the refine network
               
        if checkpoint_file:
            # Initialize the coarse network
            self.coarse_net.initialize(session, checkpoint_file)
            # Continue from check point state
            vars_to_restore = self.refine_variables()
            restorer = tf.train.Saver(vars_to_restore)
            restorer.restore(session, checkpoint_file)
            
            
        else:
            # Initialize the coarse network
            if self.coarse_checkpoint_file is None:
                self.coarse_net.initialize(session)
            else:
                self.coarse_net.initialize(session, self.coarse_checkpoint_file)
            # Default initialization
            self.__init_refine(session)
    
    def __init_refine(self,session):
        refine_vars = self.refine_variables()
        init_op = tf.initialize_variables(refine_vars)
        session.run(init_op)
        
    def __add_horizontal_units(self,i,inp_h):
        with tf.variable_scope('horz'):
            num_inps = self.km/int(2**i)
            if i == 0:  
                nhu1,nhu2,crop=1024,64,0
            elif i == 1:  
                nhu1,nhu2,crop = 512,64,-2
            elif i == 2:  
                nhu1,nhu2,crop = 256,64,-4
            elif i == 3:  
                nhu1,nhu2,crop = 64,32,-8
            
            net = tf.pad(inp_h, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,nhu2 , [3, 3], stride=1,padding='VALID',
                               activation_fn=tf.nn.relu, scope='conv1')
            
            net = tf.pad(inp_h, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps , [3, 3], stride=1,padding='VALID',
                               activation_fn=tf.nn.relu, scope='conv2')
            
            net = tf.pad(inp_h, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps/2 , [3, 3], stride=1,padding='VALID',
                               activation_fn=None, scope='conv3')
        
        return net
        
    def __add_vertical_units(self,i,inp_v):
        with tf.variable_scope('vert'):
            num_inps = self.km/int(2**i)
            net = tf.pad(inp_v, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps , [3, 3], stride=1,padding='VALID',
                               activation_fn=tf.nn.relu, scope='conv1')
            net = tf.pad(inp_v, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
            net = slim.conv2d(net,num_inps/2 , [3, 3], stride=1,padding='VALID',
                               activation_fn=None, scope='conv2')

        return net
    
    def __add_refinement_module(self,i,inp_h,inp_v):
        with tf.variable_scope('refinement_%d' % (i + 1)):
            hnet = self.__add_horizontal_units(i,inp_h)
            vnet = self.__add_vertical_units(i,inp_v)
            
            # Upsample 
            net = tf.add(hnet,vnet)
            net = tf.nn.relu(net)
            shape = net.get_shape().as_list()
            target_shape = [shape[1]*2,shape[2]*2]
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

        if self.stop_horizontal_grads:
            hlist = [tf.stop_gradient(h) for h in hlist]

        return hlist
    
    def __add_sharp_mask_modules(self,net,end_points,weight_decay=0.0001):

        with tf.variable_scope(REFINE_SCOPE) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.flatten],
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                            outputs_collections=end_points_collection):
                # Horizontal output layers from the train_coarse_net net
                inp_h_list = self.__horizontal_input_tensors(end_points)
                
                #Last output tensor ... 512-d 
                top_layer = self.__top_512d_tensor(end_points)
                
                net = slim.fully_connected(top_layer,int(self.fsz*self.fsz*self.km),activation_fn = None,scope='sharp_linear')
                net = tf.reshape(net,[-1,self.fsz,self.fsz,self.km],name='sharp_reshape')
                for i in range(4):
                    net =  self.__add_refinement_module(i,inp_h_list[i], net) 
                
                net = tf.pad(net, [[0,0],[1,1],[1,1],[0,0]],mode = "SYMMETRIC")
                net = slim.conv2d(net,1 , [3, 3], stride=1,padding='VALID',
                               activation_fn=None, scope='conv')    
            end_points = dict(tf.get_collection(end_points_collection))
        return net,end_points
            
            
if __name__ == '__main__':
    inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
    refine_net = RefineNet(inp,RESNET_50,'exp/test3/iters-24309')
    net,end_points = refine_net.net,refine_net.end_points
    print("Done")
