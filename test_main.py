'''
Created on Dec 1, 2016

@author: george
'''
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
#from ops.segnet_loss import weighted_per_image_loss
#from ops.segnet_loss import weighted_per_image_loss2
#from ops.segnet_loss import dist_loss
import os, sys
import numpy as np
import math
from datetime import datetime
import time
from math import ceil
from tensorflow.python.ops import gen_nn_ops
import skimage
import skimage.io
import re
from models import resnet_v1
from net import segnet3 as segnet
slim = tf.contrib.slim

# modules

IMG_HEIGHT = 360
IMG_WIDTH = 480




if __name__ == '__main__':
    inp = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 7], name='input')
    label = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH], name='label')
    weights = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH], name='weights')
    keep_prob = tf.placeholder(tf.float32)
    is_training_pl = tf.placeholder(tf.bool, name="segnet_is_training")

    logits = segnet.inference(inp,is_training_pl)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        segnet.initialize_resnet(sess)
    print (logits)
