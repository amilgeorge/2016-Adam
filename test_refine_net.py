'''
Created on Oct 15, 2016

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
from net.coarsenet import CoarseNet

NETWORK = 'coarse'
CHECKPOINT_1 = 'exp/refine-test_1/iters-38609'
CHECKPOINT_2 = 'exp/r-coarse-weighted-f5-5-4/iters-6000'
CHECKPOINT_3 = 'exp/r-c-ch7-3-1/iters-6000'
CHECKPOINT_4 = 'exp/c-ch7-aug-drop-1/iters-17000'
CHECKPOINT_5 = 'exp/c-ch7-aug-drop-reg-3/iters-10000'
CHECKPOINT_6 = 'exp/c-ch7-aug-drop-reg-4/iters-4000'

CHECKPOINT_18 = 'exp/c-ch7-aug4-weightedlossP3-3/iters-30000'

CHECKPOINT = CHECKPOINT_18
davis = DataAccessHelper()
logger = getLogger() 


def threshold_image(prediction,threshold=0.5):
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    return prediction

def save_image(out_dir,sequence_name,frame_no,img):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    io.imsave(out_path, img)
    
def build_test_model():
    inp = tf.placeholder(tf.float32,shape=[None,224,224,7],name='input')
    if NETWORK =="coarse" :
        net = CoarseNet(inp,is_training=tf.placeholder(tf.bool))
    else:
        net = RefineNet(inp) 
    
    return net
    
def test_sequence(session,net,sequence_name,out_dir,keep_size = True):
    
    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path, [224,224])*255
    
    for frame_no in range(min(frames),max(frames)+1):
        #Prepare input
        image_path = davis.image_path(sequence_name, frame_no)
        inp_img = sampleinputprovider.prepare_input_ch7(image_path, prev_mask)
        
        # Run model
        prediction = net.im_predict(session,inp_img)
        
        # Prepare mask for next iteration
        pred_mask = threshold_image(prediction)
        
        pred_upscaled = transform.resize(prediction,[224,224])
        pred_mask_upscaled = threshold_image(pred_upscaled)
        prev_mask = pred_mask_upscaled * 255
        
        if keep_size:
            img_shape = davis.image_shape(image_path)[0:2]
            pred_mask = transform.resize(pred_mask, img_shape)
        
        save_image(out_dir, sequence_name, frame_no, skimage.img_as_ubyte(pred_mask))

def test_net(sequences,out_dir):
    #if sequences == None:
    #    sequences=[name for name in os.listdir(inp_dir) if os.path.isdir(name)]
    
    net = build_test_model()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        net.initialize(sess, CHECKPOINT)
        
        for seq in sequences:
            logger.info('Testing sequence: {}'.format(seq))
            test_sequence(sess, net, seq,out_dir)
        
if __name__ == '__main__':
    test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
    m=re.match(r"exp/(.*)/iter.*",CHECKPOINT)
    res_dir = m.group(1)+"-"+NETWORK + "-1"
    out_dir = "../Results/{}/480p".format(res_dir)
    logger.info("Output to: {}",out_dir)	
    test_net(test_sequences, out_dir=out_dir)
