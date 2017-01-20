'''
Created on Dec 1, 2016

@author: george
'''

import os
import tensorflow as tf
from net.refinenet import RefineNet
from skimage import io,transform
from dataprovider import sampleinputprovider
from dataprovider.davis import DataAccessHelper
from common.logger import getLogger
import skimage
import re
from net import segnet2 as segnet
from dataprovider.finetuneinputprovider import FineTuneDataProvider
from common.diskutils import ensure_dir
import numpy as np
import matplotlib.pyplot as plt

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
CHECKPOINT_19 = 'exp/segnet-res2-ch7-aug-ldiff-wl-P10N1-1/iters-18500' # ldiff
CHECKPOINT_20 = 'exp/segnet-res2-ch7-aug-wl-P3N1-1/iters-30000'



CHECKPOINT = CHECKPOINT_19
learn_changes_mode = False
use_gt_prev_mask = True

davis = DataAccessHelper()
logger = getLogger() 

RUN_ID = "f-segnet-ch7-aug-10"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480

def threshold_image(prediction,threshold=0.5):
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    return prediction

def save_image(out_dir,sequence_name,frame_no,img):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    io.imsave(out_path, img)

def save_feature_maps(out_dir,sequence_name,frame_no,fmaps):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    numfmaps = fmaps.shape[2]
    nrows = numfmaps/8
    ncols = 8
    f = plt.figure(figsize=(30,30))
    for i in range(numfmaps):
        plt.subplot(nrows,ncols,i+1)
        plt.imshow(fmaps[:,:,i])
        plt.axis('off')
        #plt.colorbar()

    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    plt.savefig(out_path,bbox_inches='tight')
    plt.close(f)
    
def build_test_model():
    inp = tf.placeholder(tf.float32,shape=[None,IMAGE_HEIGHT,IMAGE_WIDTH,7],name='input')
    is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")
    logit = segnet.inference(inp,labels=None, phase_train=is_training_pl)

    #all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)#tf.all_variables()
    #for v in all_vars:
    #    print(v.name)
    #print(pool1.name)

    pool1 = tf.get_default_graph().get_tensor_by_name("pool1:0")
    decode1 = tf.get_default_graph().get_tensor_by_name("conv_decode1/cond/Merge:0")


    #out=tf.reshape(tf.nn.softmax(logit),[-1,224,224,2])

    logit = tf.reshape(logit, (-1, segnet.NUM_CLASSES))
    out = tf.reshape(tf.nn.softmax(logit),[-1,IMAGE_HEIGHT,IMAGE_WIDTH,segnet.NUM_CLASSES])
    ret_val = {"inp_pl":inp,
               "out" : out,
               "is_training_pl":is_training_pl,
               "pool1":pool1,
                "decode1":decode1}
    return ret_val
    
def test_sequence(session,net,sequence_name,out_dir,keep_size = True):
    
    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path, [IMAGE_HEIGHT,IMAGE_WIDTH])*255
    save_image(out_dir, sequence_name, min(frames), davis.read_label(label_path))

    for frame_no in range(min(frames)+1,max(frames)+1):
        #Prepare input
        image_path = davis.image_path(sequence_name, frame_no)
        inp_img = sampleinputprovider.prepare_input_ch7(image_path, prev_mask)
        
        # Run model
        #prediction = net.im_predict(session,inp_img)
        inp = net['inp_pl']
        is_training_pl = net['is_training_pl']
        out = net['out']
        pool1 = net['pool1']
        decode1 = net['decode1']
        result = session.run([out,pool1,decode1],
                             feed_dict={inp:inp_img,
                             is_training_pl:False})
        prediction = result[0][0, :, :, 1]
        pool1fmaps = result[1][0,:,:,:]
        decode1fmaps = result[2][0,:,:,:]

        #print(result[0][0,1,1,1],result[0][0,1,1,0])

        if(learn_changes_mode):
            bin_prev_mask = prev_mask ==255
            pred_changes =  threshold_image(prediction).astype(bool)
            pred_mask = np.logical_xor(bin_prev_mask,pred_changes).astype(float)

        else:
            # Prepare mask for next iteration
            pred_mask = threshold_image(prediction)
        
        if use_gt_prev_mask :
            prev_label_path = davis.label_path(sequence_name, frame_no - 1)
            prev_mask = davis.read_label(prev_label_path, [IMAGE_HEIGHT, IMAGE_WIDTH]) * 255
        else:
            prev_mask = pred_mask * 255
        
        if keep_size:
            img_shape = davis.image_shape(image_path)[0:2]
            print(img_shape)
            print(pred_mask.shape,pred_mask.dtype)
            pred_mask = transform.resize(pred_mask, img_shape)
        
        #save_image(out_dir, sequence_name, frame_no, skimage.img_as_ubyte(pred_mask))
        save_feature_maps(os.path.join(out_dir,'pool1'), sequence_name, frame_no,pool1fmaps)
        save_feature_maps(os.path.join(out_dir,'decode1'), sequence_name, frame_no,decode1fmaps)


def test_net(sequences,out_dir):
    #if sequences == None:
    #    sequences=[name for name in os.listdir(inp_dir) if os.path.isdir(name)]
    
    net = build_test_model()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        restorer2 = tf.train.Saver()
        restorer2.restore(sess,CHECKPOINT)
        
        for seq in sequences:
            logger.info('Testing sequence: {}'.format(seq))
            test_sequence(sess, net, seq,out_dir)


if __name__ == '__main__':
    #test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
    test_sequences = ['bear']
    m=re.match(r"exp/(.*)/iter.*",CHECKPOINT)
    res_dir = m.group(1)
    if(use_gt_prev_mask):
        res_dir = res_dir+'-gtmask'

    res_dir = res_dir +"-1"

    out_dir = "../Results/{}/feature_maps".format(res_dir)
    logger.info("Output to: {}".format(out_dir))
    #build_test_model()
    test_net(test_sequences, out_dir=out_dir)
