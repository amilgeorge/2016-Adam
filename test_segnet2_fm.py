'''
Created on Dec 1, 2016

@author: george
'''

import os
import tensorflow as tf
from net.refinenet import RefineNet
from skimage import io,transform,morphology
#from dataprovider import sampleinputprovider
from dataprovider import inputhelper
from dataprovider.preprocess import vgg_preprocess
from dataprovider import frame_no_calculator as fnc
import matplotlib.pyplot as plt


from dataprovider.davis import DataAccessHelper
from common.logger import getLogger
import skimage
import re
#from net import segnet2 as segnet
from net import segnet_brn as segnet

#from dataprovider.finetuneinputprovider import FineTuneDataProvider
from common.diskutils import ensure_dir
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
TESTPARAMS49 = ('exp/segnet480pvgg-wl-dp2-osvos-val-O0-3/iters-45000',0)
TESTPARAMS50 = ('exp/segnet480pvgg-d2017-wl-O1-mo2/iters-45000',1)
TESTPARAMS51 = ('exp_repo/s480pvgg-segnet_brn-daviscombo-O1-Plabel_to_dist-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iters-685000',1)
TESTPARAMS52 = ('exp_repo/s480pvgg-segnet_brn-daviscombo-O1-Plabel_to_dist-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iters-685000',5)
TESTPARAMS53 = ('exp_repo/s480pvgg-segnet_brn-davis2016-O1-PNone-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iters-600000',1)
TESTPARAMS54 = ('exp_repo/s480pvgg-segnet_brn-davis2016-O1-Plabel_to_dist-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iters-600000',1)

CHECKPOINT = None
OFFSET  = None
DILATE_SZ = 0
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

def save_feature_maps(out_dir,sequence_name,frame_no,fmaps):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    numfmaps = fmaps.shape[2]
    nrows = numfmaps/8
    ncols = 8
    f = plt.figure(figsize=(30,30))
    for i in range(numfmaps):
        ax = plt.subplot(nrows,ncols,i+1)
        im = ax.imshow(fmaps[:,:,i])
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #plt.colorbar()

    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    plt.savefig(out_path,bbox_inches='tight')
    plt.close(f)

def save_feature_maps_mean(out_dir,sequence_name,frame_no,img,pred,mask,fmap):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    f = plt.figure(figsize=(8,8))
    plt.subplot(2, 2, 1)
    plt.imshow(np.uint8(img))

    plt.subplot(2, 2, 2)
    #fmap_resized = transform.resize(fmap, [IMAGE_HEIGHT,IMAGE_WIDTH])

    plt.imshow(pred)

    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(fmap)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(mask)



    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    plt.tight_layout()
    plt.savefig(out_path,bbox_inches='tight')
    plt.close(f)

def prev_mask_path(out_dir,seq_name, frame_no, pfc):
    prev_frame_no = pfc(frame_no)
    return mask_path(out_dir,seq_name,prev_frame_no)

def osvos_mask_path(seq_name, frame_no, pfc):
    prev_frame_no = pfc(frame_no)

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

    logit,endpt = segnet.inference_vgg16_withskip_withendpoints(inp,labels=None, phase_train=is_training_pl)
    #logit = segnet.inference_vgg16_withdrop(inp,labels=None, phase_train=is_training_pl,keep_prob = keep_prob)
    #out=tf.reshape(tf.nn.softmax(logit),[-1,224,224,2])
    logit = tf.reshape(logit, (-1, segnet.NUM_CLASSES))
    out = tf.reshape(tf.nn.softmax(logit),[-1,IMAGE_HEIGHT,IMAGE_WIDTH,segnet.NUM_CLASSES])
    ret_val = {"inp_pl":inp,
               "out" : out,
               "is_training_pl":is_training_pl,
               "decode1_1":endpt['decode1_1']
               }
    return ret_val
    
def test_sequence(session,net,sequence_name,out_dir,pfc,prev_mask_dir=None,prev_mask_preprocessor = None):

    mask_out_dir = os.path.join(out_dir,'480p')
    decode1_1_dir = os.path.join(out_dir, 'decode1_1')
    decode1_1_mean_dir = os.path.join(out_dir, 'decode1_1_mean')

    if prev_mask_dir is None:
        prev_mask_dir = mask_out_dir
    prob_map_dir = os.path.join(out_dir,'prob_maps')
    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path, [IMAGE_HEIGHT,IMAGE_WIDTH])*255
    save_image(mask_out_dir, sequence_name, min(frames), skimage.img_as_ubyte(davis.read_label(label_path, [IMAGE_HEIGHT,IMAGE_WIDTH])))


    for frame_no in range(min(frames)+1,max(frames)+1):
        #Prepare input
        image_path = davis.image_path(sequence_name, frame_no)
        prev_frame_no = pfc(frame_no)
        prev_image_path = davis.image_path(sequence_name, prev_frame_no)

        img = davis.read_image(image_path)
        prev_img = davis.read_image(prev_image_path)

        if use_gt_prev_mask:
            prev_label_path = davis.label_path(sequence_name, prev_frame_no)
            prev_mask = davis.read_label(prev_label_path, [IMAGE_HEIGHT, IMAGE_WIDTH]) * 255
        else:

            prev_label_path = prev_mask_path(prev_mask_dir, sequence_name, frame_no , pfc)

            print("using prev mask {} for frame {}".format(prev_label_path,frame_no))
            prev_mask = read_label(prev_label_path, [IMAGE_HEIGHT, IMAGE_WIDTH])
            prev_mask = threshold_image(prev_mask)
            assert np.logical_or((prev_mask == 1), (prev_mask == 0)).all(), "expected 0 or 1 in binary mask"
            if prev_mask_preprocessor is not None:
                    prev_mask = inputhelper.prev_mask_preprocess(prev_mask,prev_mask_preprocessor)

            prev_mask = prev_mask * 255
            if DILATE_SZ > 0:
                print("dilating")
                prev_mask = morphology.dilation(prev_mask, np.ones([DILATE_SZ, DILATE_SZ]))
            #print("dilating")
            #prev_mask = morphology.dilation(prev_mask, np.ones([10, 10]))
            #assert np.logical_or((prev_mask == 255), (prev_mask == 0)).all(), "expected 0 or 255 in binary mask"


        inp_img = inputhelper.prepare_input_img_uint8(img, prev_mask, prev_img)
        inputhelper.verify_input_img(inp_img[0,:,:,:],prev_mask_preprocessor)
        inp_img = vgg_preprocess(inp_img)

        # Run model
        #prediction = net.im_predict(session,inp_img)
        inp = net['inp_pl']
        is_training_pl = net['is_training_pl']
        #keep_prob = net['keep_prob']
        out = net['out']
        decode1_1 = net['decode1_1']
        result = session.run([out,decode1_1],
                             feed_dict={inp:inp_img,
                             is_training_pl:False})
        prediction = result[0][0,:,:,1]
        decode1_1_fm = result[1][0, :, :, :]
        decode1_1_fm_mean = np.mean(np.square(decode1_1_fm), axis=2)

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
        

        save_feature_maps(decode1_1_dir,sequence_name,frame_no,decode1_1_fm)
        save_feature_maps_mean(decode1_1_mean_dir, sequence_name, frame_no,img,prediction,
                               skimage.img_as_ubyte(pred_mask),decode1_1_fm_mean)
        save_image(mask_out_dir, sequence_name, frame_no, skimage.img_as_ubyte(pred_mask))

def test_network(sess,net,out_dir,pfc,prev_mask_dir=None,prev_mask_preprocessor = None):
    test_sequences = davis.test_sequence_list() + davis.train_sequence_list()
    for seq in test_sequences:
        logger.info('Testing sequence: {}'.format(seq))
        test_sequence(sess, net, seq, out_dir, pfc,prev_mask_dir,prev_mask_preprocessor)


def test_net(sequences,out_dir,pfc,prev_mask_dir=None,prev_mask_preprocessor=None):
    #if sequences == None:
    #    sequences=[name for name in os.listdir(inp_dir) if os.path.isdir(name)]
    
    net = build_test_model()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        restorer2 = tf.train.Saver()
        restorer2.restore(sess,CHECKPOINT)
        for seq in sequences:
            test_sequence(sess, net, seq, out_dir, pfc, prev_mask_dir, prev_mask_preprocessor)


if __name__ == '__main__':
    #global CHECKPOINT
    #global OFFSET

    test_points = [TESTPARAMS54]
    prev_mask_preprocessor =inputhelper.PREPROCESS_LABEL_TO_DIST

    for tp in test_points:
        CHECKPOINT = tp[0]
        OFFSET = tp[1]

        pfc = fnc.get(fnc.POLICY_OFFSET,OFFSET)

        #test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
        test_sequences = ['car-roundabout']
        m=re.match(r"exp.*/(.*)/iter.*",CHECKPOINT)
        res_dir = m.group(1)

        print ("Testing for {}".format(res_dir) )

        if(use_gt_prev_mask):
            res_dir = res_dir+'-gtmask'


        res_dir = res_dir+'-O{}'.format(OFFSET)

        res_dir = res_dir +"-d0-1"

        out_dir = "../Results/{}".format(res_dir)
        logger.info("Output to: {}".format(out_dir))
        test_net(test_sequences, out_dir=out_dir,pfc=pfc,prev_mask_preprocessor=prev_mask_preprocessor)
