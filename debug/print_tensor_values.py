'''
Created on Oct 11, 2016

@author: george
'''
import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from net.coarsenet import CoarseNet
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.learn.python.learn.utils import checkpoints

inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
coarse_net = CoarseNet(inp,'resnet_v1_50',False)
net,end_points = coarse_net.net,coarse_net.end_points
 
 
 # Attach sigmoid and reshape
coarse_out= tf.reshape(tf.sigmoid(net),[-1,56,56,1])
sess = tf.InteractiveSession()
global_step_var = tf.Variable(0, trainable=False)
 
coarse_net.initialize(sess,'exp/coarse-weighted-f5-5/iters-12000')

with tf.variable_scope("resnet_v1_50") as scope:
    #tf.get_variable_scope().reuse = True
    scope.reuse_variables()
    v = tf.get_variable('conv1/BatchNorm/moving_mean')
    print(v.eval())
    
 
print("end")