'''
Created on Dec 1, 2016

@author: george
'''
#import tensorflow as tf
#from tensorflow.python.framework import ops
#from tensorflow.python.framework import dtypes
#from ops.segnet_loss import weighted_per_image_loss
#from ops.segnet_loss import weighted_per_image_loss2
#from ops.segnet_loss import dist_loss
import os, sys
import numpy as np
import math
from datetime import datetime
import time
from math import ceil
#from tensorflow.python.ops import gen_nn_ops
import skimage
import skimage.io
import re
#from models import resnet_v1
#from net import segnet3 as segnet
#slim = tf.contrib.slim
from dataprovider.davis_cached_2016 import DataAccessHelper
from dataprovider import inputhelper as ih

# modules

IMG_HEIGHT = 360
IMG_WIDTH = 480

from dataprovider.imdbdataprovider import InputProvider
from dataprovider import frame_no_calculator as fnc
from dataprovider import imdb

if __name__ == '__main__':
    """
    inp = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 7], name='input')
    label = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH], name='label')
    weights = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH], name='weights')
    keep_prob = tf.placeholder(tf.float32)
    is_training_pl = tf.placeholder(tf.bool, name="segnet_is_training")

    logits = segnet.inference(inp,is_training_pl)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        segnet.initialize_resnet(sess)
    print (logits)
    """
    davis = DataAccessHelper()

    pfc = fnc.get(fnc.POLICY_OFFSET, 1)
    inp_provider = InputProvider(db_name=imdb.IMDB_CUSTOM_MASK_DAVIS2016, prev_frame_calculator=pfc)
    inp_provider.db.set_policy(imdb.CustomMaskDB2016.POLICY_SELECT_CM,0.85)
    inp_provider.db.set_mask_folder("/usr/stud/george/workspace/adam/test_out/s480pvgg-daviscombo-O1-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-3/iter-50000/480p")
    trainseqs = davis.train_sequence_list()
    for seq in trainseqs:
        all_frames = davis.all_frames_nums(seq)
        for frame_no in all_frames:
            inp_provider.db.get_selected_mask(seq,frame_no)
    inp_provider.db.set_mask_folder("/usr/stud/george/workspace/adam/test_out/s480p-custommaskdavis2016-O1-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-3/iter-54000/480p")
    for seq in trainseqs:
        all_frames = davis.all_frames_nums(seq)
        for frame_no in all_frames:
            inp_provider.db.get_selected_mask(seq,frame_no)

    print(inp_provider.db.disp_stats())
    print("Done")
    
    """
    davis = DataAccessHelper(load_cached = False)

    img1_path = davis.image_path('bear', 0)
    img2_path = davis.image_path('blackswan', 0)

    img1 = davis.read_image_disk(img1_path)
    img2 =davis.read_image_disk(img2_path)

    stked = np.concatenate((np.expand_dims(img1,0), np.expand_dims(img2,0)),axis=0)
    stked_bgr = stked[:,:,:,::-1]
    bgr1 = stked[0, :, :, [2, 1, 0]]

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', stked[0,:,:,[2,1,0]])
    cv2.waitKey()
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.imshow('image2', stked[1,:,:,[2,1,0]])
    cv2.waitKey()
"""




