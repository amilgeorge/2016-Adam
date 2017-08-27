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

from net import mergenets
from dataprovider.davis import DataAccessHelper
from common.logger import getLogger
import skimage
import re
from net import segnet2 as segnet
import matplotlib.pyplot as plt
#from dataprovider.finetuneinputprovider import FineTuneDataProvider
from common.diskutils import ensure_dir
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

NETWORK = 'coarse'

TESTPARAMS40 = ('exp/segnetvggwithskip-wl-distosvos-O1-1/iters-45000',1)
TESTPARAMS41 = ('exp/segnetvggwithskip-wl-distosvos-O5-1/iters-45000',5)
TESTPARAMS47 = ('exp/segnet480pvgg-wl-dp2-osvos-O1-1/iters-45000',1)
TESTPARAMS48 = ('exp/segnet480pvgg-wl-dp2-osvos-lr001-O1-1/iters-30000',1)
TESTPARAMS49 = ('exp/segnet480pvgg-wl-dp2-osvos-val-O0-3/iters-45000',0)
TESTPARAMS50 = ('exp/segnet480pvgg-d2017-wl-O1-mo2/iters-45000',1)
TESTPARAMS51 = ('exp/mergeosvosnet-v1_baseline-seqdavis2016-B1O-1-adam<1e-6>-opt-adam-<1e-4>-250iter-1/iters-150000',1)



CHECKPOINT = None
OFFSET  = None
DILATE_SZ = 0
learn_changes_mode = False
use_gt_prev_mask = False
use_custom_prev_mask_folder = True

SAVE_PREDICTIONS = False

davis = DataAccessHelper()
logger = getLogger()

RUN_ID = "f-segnet-ch7-aug-10"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 854

def reinit_branch(session,seq):
    """
    Re init osvos branch with fine tuned checkpoint for sequence
    :param session:
    :param seq:
    :return:
    """

    checkpoint_file = os.path.join('..','OSVOS-TensorFlow','models',seq,'{}.ckpt-2000'.format(seq))
    mergenets.initialize_merge_net(session, None, checkpoint_file)

def prev_mask_path(out_dir,seq_name, frame_no, pfc):
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
    keep_prob = tf.placeholder(tf.float32)

    logit,endpt = mergenets.inference_merge_two_branch_baseline(inp, phase_train=is_training_pl,return_branches=True)
    #logit = segnet.inference_vgg16_withdrop(inp,labels=None, phase_train=is_training_pl,keep_prob = keep_prob)
    #out=tf.reshape(tf.nn.softmax(logit),[-1,224,224,2])
    logit = tf.reshape(logit, (-1, segnet.NUM_CLASSES))
    out = tf.reshape(tf.nn.softmax(logit),[-1,IMAGE_HEIGHT,IMAGE_WIDTH,segnet.NUM_CLASSES])
    ret_val = {"inp_pl":inp,
               "out" : out,
               "is_training_pl":is_training_pl,
               "branch1":endpt['branch1'],
               "branch2":endpt['branch2']
               }
    return ret_val

def test_sequence(session,net,sequence_name,out_dir, pfc):

    mask_out_dir = os.path.join(out_dir,'480p')
    prob_map_dir = os.path.join(out_dir,'prob_maps')
    branch1_dir = os.path.join(out_dir, 'branch1')
    branch1_mean_dir = os.path.join(out_dir, 'branch1_mean')
    branch2_dir = os.path.join(out_dir, 'branch2')
    branch2_mean_dir = os.path.join(out_dir, 'branch2_mean')

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

            if use_custom_prev_mask_folder:
                custom_prev_mask_out_dir = os.path.join('test_out/s480pvgg-davis2016-O1-osvosold-reg1e-4-adam<1e-6>-de-1/iter-500000','480p')
                prev_label_path = prev_mask_path(custom_prev_mask_out_dir, sequence_name, frame_no , pfc)
            else:
                prev_label_path = prev_mask_path(mask_out_dir, sequence_name, frame_no , pfc)

            print("using prev mask {} for frame {}".format(prev_label_path,frame_no))
            prev_mask = read_label(prev_label_path, [IMAGE_HEIGHT, IMAGE_WIDTH])
            prev_mask = threshold_image(prev_mask)


            assert np.logical_or((prev_mask == 1), (prev_mask == 0)).all(), "expected 0 or 1 in binary mask"


            prev_mask = prev_mask * 255
            if DILATE_SZ > 0:
                print("dilating")
                prev_mask = morphology.dilation(prev_mask, np.ones([DILATE_SZ, DILATE_SZ]))
            #print("dilating")
            #prev_mask = morphology.dilation(prev_mask, np.ones([10, 10]))
            assert np.logical_or((prev_mask == 255), (prev_mask == 0)).all(), "expected 0 or 255 in binary mask"


        inp_img = inputhelper.prepare_input_img_uint8(img, prev_mask, prev_img)
        inputhelper.verify_input_img(inp_img[0,:,:,:])
        inp_img = vgg_preprocess(inp_img)

        # Run model
        #prediction = net.im_predict(session,inp_img)
        inp = net['inp_pl']
        is_training_pl = net['is_training_pl']
        #keep_prob = net['keep_prob']
        out = net['out'],
        branch1 = net['branch1']
        branch2 = net['branch2']

        result = session.run([out,branch1,branch2],
                             feed_dict={inp:inp_img,
                             is_training_pl:False})
        print(result[0][0].shape)
        prediction = result[0][0][0,:,:,1]
        branch1_fm = result[1][0,:,:,:]
        branch2_fm = result[2][0,:,:,:]
        branch1_fm_mean = np.mean(np.square(branch1_fm),axis=2)
        branch2_fm_mean = np.mean(np.square(branch2_fm),axis=2)


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



        save_image(mask_out_dir, sequence_name, frame_no, skimage.img_as_ubyte(pred_mask))
        """
        save_feature_maps_mean(branch1_mean_dir, sequence_name, frame_no,img,prediction,skimage.img_as_ubyte(pred_mask),branch1_fm_mean)
        save_feature_maps_mean(branch2_mean_dir, sequence_name, frame_no,img,prediction,skimage.img_as_ubyte(pred_mask),branch2_fm_mean)

        save_feature_maps(branch1_dir,sequence_name,frame_no,branch1_fm)
        save_feature_maps(branch2_dir,sequence_name,frame_no,branch2_fm)
        """

def test_network(sess,net,out_dir,pfc):
    test_sequences = davis.test_sequence_list() + davis.train_sequence_list()
    for seq in test_sequences:
        logger.info('Testing sequence: {}'.format(seq))
        reinit_branch(sess,seq)
        test_sequence(sess, net, seq, out_dir, pfc)


def test_net(sequences,out_dir,pfc):
    #if sequences == None:
    #    sequences=[name for name in os.listdir(inp_dir) if os.path.isdir(name)]

    net = build_test_model()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        restorer2 = tf.train.Saver()
        restorer2.restore(sess,CHECKPOINT)

        for seq in sequences:
            logger.info('Testing sequence: {}'.format(seq))
            reinit_branch(sess, seq)
            test_sequence(sess, net, seq,out_dir, pfc)


if __name__ == '__main__':
    #global CHECKPOINT
    #global OFFSET

    test_points = [TESTPARAMS51]

    for tp in test_points:
        CHECKPOINT = tp[0]
        OFFSET = tp[1]

        pfc = fnc.get(fnc.POLICY_OFFSET,OFFSET)

        test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
        #bmx-trees
        #test_sequences = ['car-roundabout']
        m=re.match(r"exp/(.*)/iter.*",CHECKPOINT)
        res_dir = m.group(1)

        print ("Testing for {}".format(res_dir) )

        if(use_gt_prev_mask):
            res_dir = res_dir+'-gtmask'


        res_dir = res_dir+'-O{}'.format(OFFSET)

        res_dir = res_dir +"-1"
        if use_custom_prev_mask_folder:
            res_dir = res_dir + "-cmask"

        out_dir = "../Results/{}".format(res_dir)
        logger.info("Output to: {}".format(out_dir))
        test_net(test_sequences, out_dir=out_dir,pfc=pfc)
