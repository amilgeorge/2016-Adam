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
#from dataprovider.inputprovider import InputProvider
from dataprovider import imgprovider
from common import diskutils
from dataprovider.preprocess import vgg_preprocess, reverse_vgg_preprocess

slim = tf.contrib.slim

# modules

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480

if __name__ == '__main__':
    imgprovider.test_gen_samples()
