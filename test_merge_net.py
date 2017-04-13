'''
Created on Dec 1, 2016

@author: george
'''

import os
import tensorflow as tf
from net.refinenet import RefineNet
from skimage import io,transform
#from dataprovider import sampleinputprovider
from dataprovider import imgprovider

from dataprovider.davis import DataAccessHelper
from common.logger import getLogger
import skimage
import re
from net import segnet2 as segnet
from dataprovider.finetuneinputprovider import FineTuneDataProvider
from common.diskutils import ensure_dir
import numpy as np

NETWORK = 'coarse'
CHECKPOINT_1 = 'exp/refine-test_1/iters-38609'
CHECKPOINT_2 = 'exp/r-coarse-weighted-f5-5-4/iters-6000'
CHECKPOINT_3 = 'exp/r-c-ch7-3-1/iters-6000'
CHECKPOINT_4 = 'exp/c-ch7-aug-drop-1/iters-17000'
CHECKPOINT_5 = 'exp/c-ch7-aug-drop-reg-3/iters-10000'
CHECKPOINT_6 = 'exp/c-ch7-aug-drop-reg-4/iters-4000'
CHECKPOINT_7 = 'exp/segnet-ch7-4/iters-30000'
CHECKPOINT_8 = 'exp/segnet-ch7-6/bkp/iters-7000'
CHECKPOINT_9 = 'exp/segnet-ch7-8/iters-5000'
CHECKPOINT_10 = 'exp/segnet-ch7-9/iters-8000'
CHECKPOINT_11 = 'exp/segnet-ch7-aug-10/iters-9000'
CHECKPOINT_12 = 'exp/segnet-ch7-aug-dynWeightedloss-2/iters-9500'
CHECKPOINT_13 = 'exp/segnet-ch7-aug-Weightedloss01-1/iters-19500'
CHECKPOINT_14 = 'exp/segnet-ch7-aug-Weightedloss05-1/iters-19500'
CHECKPOINT_15 = 'exp/segnet-ch7-aug-Weightedloss05-1/iters-10000'
CHECKPOINT_16 = 'exp/segnet-ch7-aug-WeightedlossP10-1/iters-19500'
CHECKPOINT_17 = 'exp/segnet-ch7-aug-WeightedlossP3-1/iters-19500'
CHECKPOINT_18 = 'exp/segnet-ch7-aug-dwl-1/iters-30000'
CHECKPOINT_19 = 'exp/segnet-res2-ch7-aug-wl-P10N1-1/iters-18500' # ldiff
CHECKPOINT_20 = 'exp/segnet-res2-ch7-aug-wl-P3N1-1/iters-30000'
CHECKPOINT_21 = 'exp/segnetvgg-res2-ch7-aug-wl-osvos-1/iters-30000'
CHECKPOINT_22 = 'exp/segnetvgg-res2-ch7-aug-wl-osvos-7/iters-30000'
CHECKPOINT_23 = 'exp/segnetvggwithskip-res2-ch7-aug-wl-osvos-1/iters-30000'
CHECKPOINT_24 = 'exp/segnetvggwithskip-res2-ch7-aug-wl-P3N1-1/iters-30000'
CHECKPOINT_25 = 'exp/segnetvggwithskip-res2-ch7-aug-wl-ldiffosvos-1/iters-30000'
CHECKPOINT_26 = ('exp/segnetvggwithskip-res2-ch7-aug-wl-osvos-O5-1/iters-45000',5)
CHECKPOINT_27 = 'exp/segnetvggwithskip-res2-ch7-aug-wl-osvos-O1O6-1/iters-45000'
CHECKPOINT_28 = 'exp/segnetvggwithskip-res2-ch7-aug-wl-osvos-O10-1/iters-45000'
CHECKPOINT_29 = 'exp/segnetvggwithskip-res2-ch7-aug-wl-dist-O5-1/iters-36000'
CHECKPOINT_30 = 'exp/segnetvggwithdrop-wl-osvos-O1-1/iters-45000'
CHECKPOINT_31 = 'exp/segnetvggwithdrop-half-wl-osvos-O1-1/iters-45000'
CHECKPOINT_32 = 'exp/segnetvggwithdrop-half2-wl-osvos-O1-1/iters-90000'
CHECKPOINT_33 = 'exp/segnetvggwithdrop-wl-osvos-O1O6-1/iters-45000'
CHECKPOINT_34 = 'exp/segnetvggwithskip-half-wl-osvos-O1-1/iters-45000'
CHECKPOINT_35 = 'exp/segnetvggwithskip-half-wl-osvos-O10-1/iters-45000'
CHECKPOINT_36 = 'exp/segnetvggwithskip-half-wl-osvos-O5-1/iters-45000'
CHECKPOINT_37 = 'exp/segnetvggwithskip-half2-wl-osvos-O10-1/iters-90000'
CHECKPOINT_38 = 'exp/segnetvggwithskip-half2-wl-osvos-O5-1/iters-90000'
CHECKPOINT_39 = 'exp/segnetvggwithskip-half2-wl-osvos-O1-1/iters-90000'
TESTPARAMS40 = ('exp/segnetvggwithskip-wl-distosvos-O1-1/iters-45000',1)
TESTPARAMS41 = ('exp/segnetvggwithskip-wl-distosvos-O5-1/iters-45000',5)
TESTPARAMS47 = ('exp/segnet480pvgg-wl-dp2-osvos-O1-1/iters-45000',1)
TESTPARAMS48 = ('exp/segnet480pvgg-wl-dp2-osvos-lr001-O1-1/iters-30000',1)




CHECKPOINT = None
OFFSET  = None

learn_changes_mode = False
use_gt_prev_mask = False

SAVE_PREDICTIONS = False

davis = DataAccessHelper()
logger = getLogger() 

RUN_ID = "f-segnet-ch7-aug-10"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 854


def prev_mask_path(out_dir,seq_name, frame_no, offset):
    prev_frame_no = max(0,frame_no - offset)
    return mask_path(out_dir,seq_name,prev_frame_no)

def mask_path(out_dir,seq_name, frame_no):
    label_path = os.path.join(out_dir,seq_name, '{0:05}.png'.format(frame_no))
    return label_path

def read_label(label_path, resize=None):
    image = io.imread(label_path, as_grey=True)

    if resize != None:
        image = transform.resize(image, resize)

    return image

def threshold_image(prediction,threshold=0.5):
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    return prediction

def save_image(out_dir,sequence_name,frame_no,img):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    io.imsave(out_path, img)

def save_prediction(out_dir,sequence_name,frame_no,prediction):
    out_seq_dir = os.path.join(out_dir, sequence_name)
    os.makedirs(out_seq_dir, exist_ok=True)
    out_path = os.path.join(out_seq_dir, '{0:05}.npy'.format(frame_no))
    np.save(out_path, prediction)


    
def build_test_model():
    inp = tf.placeholder(tf.float32,shape=[None,IMAGE_HEIGHT,IMAGE_WIDTH,7],name='input')
    is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")
    keep_prob = tf.placeholder(tf.float32)

    logit = segnet.inference_vgg16_withskip(inp,labels=None, phase_train=is_training_pl)
    #logit = segnet.inference_vgg16_withdrop(inp,labels=None, phase_train=is_training_pl,keep_prob = keep_prob)
    #out=tf.reshape(tf.nn.softmax(logit),[-1,224,224,2])
    logit = tf.reshape(logit, (-1, segnet.NUM_CLASSES))
    out = tf.reshape(tf.nn.softmax(logit),[-1,IMAGE_HEIGHT,IMAGE_WIDTH,segnet.NUM_CLASSES])
    ret_val = {"inp_pl":inp,
               "out" : out,
               "is_training_pl":is_training_pl
               }
    return ret_val

def get_prev_mask(mask_out_dir,sequence_name,frame_no,offset):
    if offset == 0:
        prev_label_path = prev_mask_path(mask_out_dir, sequence_name, 0, 0)
    else:
        prev_label_path = prev_mask_path(mask_out_dir, sequence_name, frame_no, offset)
    print("using prev mask {} for frame {}".format(prev_label_path, frame_no))
    prev_mask = read_label(prev_label_path, [IMAGE_HEIGHT, IMAGE_WIDTH])
    prev_mask = threshold_image(prev_mask)
    assert np.logical_or((prev_mask == 1), (prev_mask == 0)).all(), "expected 0 or 1 in binary mask"
    prev_mask = prev_mask * 255
    return prev_mask

def test_sequence(session,net,sequence_name,out_dir,branch1_offset,branch2_offset,keep_size = True):

    mask_out_dir = os.path.join(out_dir,'480p')
    prob_map_dir = os.path.join(out_dir,'prob_maps')
    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path, [IMAGE_HEIGHT,IMAGE_WIDTH])*255
    save_image(mask_out_dir, sequence_name, min(frames), skimage.img_as_ubyte(davis.read_label(label_path, [IMAGE_HEIGHT,IMAGE_WIDTH])))


    for frame_no in range(min(frames)+1,max(frames)+1):
        #Prepare input
        image_path = davis.image_path(sequence_name, frame_no)

        prev_mask_branch1 = get_prev_mask(mask_out_dir,sequence_name,frame_no,branch1_offset)
        prev_mask_branch2 = get_prev_mask(mask_out_dir,sequence_name,frame_no,branch2_offset)


        inp_branch1_img = imgprovider.prepare_input_ch7(image_path, prev_mask_branch1,offset = -1*branch1_offset)
        inp_branch2_img = imgprovider.prepare_input_ch7(image_path, prev_mask_branch2,offset = -1*branch2_offset)

        # Run model
        inp_branch1_pl = net['inp_branch1_pl']
        inp_branch2_pl = net['inp_branch2_pl']
        is_training_pl = net['is_training_pl']
        #keep_prob = net['keep_prob']
        out = net['out']
        result = session.run([out], 
                             feed_dict={inp_branch1_pl:inp_branch1_img,
                                        inp_branch2_pl:inp_branch2_img,
                             is_training_pl:False})
        prediction = result[0][0,:,:,1]

        #print(result[0][0,1,1,1],result[0][0,1,1,0])
        if SAVE_PREDICTIONS:
            save_prediction(prob_map_dir, sequence_name, frame_no, prediction)

        if(learn_changes_mode):
            bin_prev_mask = prev_mask ==255
            pred_changes =  threshold_image(prediction).astype(bool)
            pred_mask = np.logical_xor(bin_prev_mask,pred_changes).astype(float)

        else:
            # Prepare mask for next iteration
            pred_mask = threshold_image(prediction)
        

        
        if keep_size:
            img_shape = davis.image_shape(image_path)[0:2]
            print(img_shape)
            print(pred_mask.shape,pred_mask.dtype)
            pred_mask = transform.resize(pred_mask, img_shape)
        
        save_image(mask_out_dir, sequence_name, frame_no, skimage.img_as_ubyte(pred_mask))

def test_network(sess,net,out_dir,branch1_offset,branch2_offset):
    test_sequences = davis.test_sequence_list() + davis.train_sequence_list()
    for seq in test_sequences:
        logger.info('Testing sequence: {}'.format(seq))
        test_sequence(sess, net, seq, out_dir, branch1_offset = branch1_offset, branch2_offset=branch2_offset)


def test_net(sequences,out_dir):
    #if sequences == None:
    #    sequences=[name for name in os.listdir(inp_dir) if os.path.isdir(name)]
    
    net = build_test_model()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        restorer2 = tf.train.Saver()
        restorer2.restore(sess,CHECKPOINT)
        
        for seq in sequences:
            logger.info('Testing sequence: {}'.format(seq))
            test_sequence(sess, net, seq,out_dir)


        
if __name__ == '__main__':
    #global CHECKPOINT
    #global OFFSET

    test_points = [TESTPARAMS48]

    for tp in test_points:
        CHECKPOINT = tp[0]
        OFFSET = tp[1]

        test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
        #test_sequences = ['paragliding-launch']
        m=re.match(r"exp/(.*)/iter.*",CHECKPOINT)
        res_dir = m.group(1)

        print ("Testing for {}".format(res_dir) )

        if(use_gt_prev_mask):
            res_dir = res_dir+'-gtmask'

        if(OFFSET >1):
            res_dir = res_dir+'-O{}'.format(OFFSET)

        res_dir = res_dir +"-iter30"

        out_dir = "../Results/{}".format(res_dir)
        logger.info("Output to: {}".format(out_dir))
        test_net(test_sequences, out_dir=out_dir)
