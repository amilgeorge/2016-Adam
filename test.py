'''
Created on Sep 26, 2016

@author: george
'''

import tensorflow as tf
import numpy as np
from skimage import io
import os
from main import build_model


import sys

inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
build_model(inp,"vgg16")
saver = tf.train.Saver()

with tf.Session() as sess:

    #saver = tf.train.import_meta_graph('metagraph.meta')
    saver.restore(sess, 'exp/test1/epoch-26.ckpt')

    print("adf")
    #saver.restore(sess, 'exp/test1/epoch-26.ckpt')
