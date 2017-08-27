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


from dataprovider.davis import DataAccessHelper
from common.logger import getLogger
import skimage
import re
#from net import segnet2 as segnet
from net import segnet_brn as segnet

#from dataprovider.finetuneinputprovider import FineTuneDataProvider
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
TESTPARAMS49 = ('exp/segnet480pvgg-wl-dp2-osvos-val-O0-3/iters-45000',0)
TESTPARAMS50 = ('exp/segnet480pvgg-d2017-wl-O1-mo2/iters-45000',1)
TESTPARAMS51 = ('exp/s480pvgg-segnet_brn-daviscombo-O1-Plabel_to_dist-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iters-5000',1)


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

def build_model_for_fine_tuning():
    inp = tf.placeholder(tf.float32,shape=[None,IMAGE_HEIGHT,IMAGE_WIDTH,7],name='input')
    label =  tf.placeholder(tf.float32,shape=[None,IMAGE_HEIGHT,IMAGE_WIDTH],name='label')
    is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")
    loss,logit = segnet.inference_vgg16_withskip(inp, label, is_training_pl)

    logit = tf.reshape(logit, (-1, segnet.NUM_CLASSES))
    out = tf.reshape(tf.nn.softmax(logit),[-1,IMAGE_HEIGHT,IMAGE_WIDTH,segnet.NUM_CLASSES])
    
    ret_val = {"inp_pl":inp,
               "label_pl":label,
               "out" : out,
               "is_training_pl":is_training_pl,
               "loss":loss }
    return ret_val
    
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
    
def test_sequence(session,net,sequence_name,out_dir,pfc,prev_mask_dir=None,prev_mask_preprocessor = None):

    mask_out_dir = os.path.join(out_dir,'480p')
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
        result = session.run([out], 
                             feed_dict={inp:inp_img,
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
        

        
        save_image(mask_out_dir, sequence_name, frame_no, skimage.img_as_ubyte(pred_mask))

def test_network(sess,net,out_dir,pfc,prev_mask_dir=None,prev_mask_preprocessor = None):
    test_sequences = davis.test_sequence_list() + davis.train_sequence_list()
    for seq in test_sequences:
        logger.info('Testing sequence: {}'.format(seq))
        test_sequence(sess, net, seq, out_dir, pfc,prev_mask_dir,prev_mask_preprocessor)


def test_net(sequences,out_dir,pfc,prev_mask_preprocessor=None):
    #if sequences == None:
    #    sequences=[name for name in os.listdir(inp_dir) if os.path.isdir(name)]
    
    net = build_test_model()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        restorer2 = tf.train.Saver()
        restorer2.restore(sess,CHECKPOINT)

        test_network(sess, net,out_dir, pfc,prev_mask_preprocessor=prev_mask_preprocessor)

def run_train_loop(seq,session,net,dp,ops,summary_writer,max_iters = 1000):
    step = 0
    saver = tf.train.Saver(max_to_keep = 50)
    exp_dir = os.path.join(EXP_DIR,seq)
    ensure_dir(exp_dir)
    ### ops ###
    apply_gradient_op = ops['train_op']
    merged_summary = ops['merged_summary']
    merged_val_summary = ops['merged_val_summary'] 
    val_loss_pl = ops['val_loss_pl'] 
    ###########
    loss = net['loss']
    inp = net['inp_pl']
    is_training_pl = net['is_training_pl']
    label = net['label_pl']
    
    
    
    def perform_validation(session,step,summary_writer):

        losses = []
        val_data = dp.val_seq_batch_itr()
        for i, sequence_batch in enumerate(val_data):
            result = session.run([loss], 
                                    feed_dict={inp:sequence_batch.images,
                                    label:sequence_batch.labels,
                                    is_training_pl:False})
            loss_value = result[0]
            losses.append(loss_value)
            logger.info('val step:{}, iters:{}, loss :{} '.format(step, i, loss_value))
            avg_loss = sum(losses)/len(losses)

            feed = {val_loss_pl: avg_loss}
            
            val_summary = session.run([merged_val_summary],feed_dict = feed)
            summary_writer.add_summary(val_summary[0],step)
    
    while step < max_iters:
        sequence_batch = dp.get_next_minibatch()
        result = session.run([apply_gradient_op, loss,merged_summary], 
                                      feed_dict={inp:sequence_batch.images,
                                                label:sequence_batch.labels,
                                                is_training_pl:True})
        summary_writer.add_summary(result[2], step )
        loss_value = result[1]
        logger.info('train iters:{} loss :{}'.format(step, loss_value))

        if step % 50 == 0:
            perform_validation(session,step,summary_writer)
        
        if step % 50 == 0:
            logger.info('Saving weights.')
            saver.save(session, os.path.join(exp_dir,'iters'),global_step = step)
            logger.info('Flushing .')                        
            summary_writer.flush()
        
        step = step+1
        
def fine_tune_net_for_seq(session,net,seq):
    
    logger.info('Starting Fine Tuning for :{}'.format(seq))
    ### ###
    events_dir = os.path.join(EVENTS_DIR,seq)
    summary_writer = tf.summary.FileWriter(events_dir, session.graph)
    #######
    ### Data Provider ###
    data_provider = FineTuneDataProvider(seq,16)
    #####################
    loss = net['loss']
    ### Optimizer ###
    optimizer = tf.train.GradientDescentOptimizer(0.001)          
    gradients = optimizer.compute_gradients(loss)                               
    apply_gradient_op = optimizer.apply_gradients(gradients)
    #################
    ### Summaries ### 
    tf.summary.scalar('/train/loss', loss)  
    val_loss_pl = tf.placeholder(tf.float32)
    VALIDATION_SUMMARIES = 'validation_summaries'
    val_loss_summary = tf.summary.scalar('/val/loss', val_loss_pl,collections=VALIDATION_SUMMARIES)  
    merged_val_summary = tf.summary.merge([val_loss_summary],collections=None)   
    merged_summary = tf.summary.merge_all()    
    #################
    
    ### Collect Ops ###
    ops = {}
    ops['train_op'] = apply_gradient_op
    ops['merged_summary'] = merged_summary
    ops['merged_val_summary'] = merged_val_summary
    ops['val_loss_pl'] = val_loss_pl
    ####################
    
    run_train_loop(seq,session,net,data_provider,ops,summary_writer)
    
    logger.info('Completed Fine Tuning for :{}'.format(seq))

    

def test_net_fine_tune(sequences,out_dir):
    for seq in sequences:
        tf.reset_default_graph()
        net = build_model_for_fine_tuning()
    
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            restorer2 = tf.train.Saver()
            restorer2.restore(sess,CHECKPOINT)
            
            fine_tune_net_for_seq(sess, net, seq)
            
            test_sequence(sess, net, seq, out_dir)
        
if __name__ == '__main__':
    #global CHECKPOINT
    #global OFFSET

    test_points = [TESTPARAMS51]

    for tp in test_points:
        CHECKPOINT = tp[0]
        OFFSET = tp[1]

        pfc = fnc.get(fnc.POLICY_OFFSET,OFFSET)

        test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
        #test_sequences = ['paragliding-launch']
        m=re.match(r"exp/(.*)/iter.*",CHECKPOINT)
        res_dir = m.group(1)

        print ("Testing for {}".format(res_dir) )

        if(use_gt_prev_mask):
            res_dir = res_dir+'-gtmask'


        res_dir = res_dir+'-O{}'.format(OFFSET)

        res_dir = res_dir +"-d10-1"

        out_dir = "../Results/{}".format(res_dir)
        logger.info("Output to: {}".format(out_dir))
        test_net(test_sequences, out_dir=out_dir,pfc=pfc,prev_mask_preprocessor=inputhelper.PREPROCESS_LABEL_TO_DIST)
