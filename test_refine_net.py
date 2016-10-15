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

COARSE_CHECKPOINT = 'asdf'
CHECKPOINT = 'exp/r-coarse-weighted-f5-5-4/iters-6000'
davis = DataAccessHelper()

def threshold_image(prediction,threshold=0.5):
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    return prediction

def save_image(out_dir,sequence_name,frame_no,img):
    out_path = os.path.join(out_dir,sequence_name,'{0:05}.png'.format(frame_no))
    io.imsave(out_path, img)
    
def build_test_model():
    inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
    refine_net = RefineNet(inp,coarse_checkpoint_file=COARSE_CHECKPOINT) 
    
    return refine_net
    
def test_sequence(session,net,sequence_name,out_dir):
    
    frames = davis.all_frames_nums(sequence_name)
    
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path, [224,224])
    
    for frame_no in range(1,max(frames)):
        image_path = davis.image_path(sequence_name, frame_no)
        img = davis.read_image(image_path, [224,224])
        img_shape = davis.image_shape(image_path)[0:2]
        inp_img = sampleinputprovider.prepare_input(img, prev_mask)
        prediction = net.im_predict(session,inp_img)
        pred_mask = threshold_image(prediction)
        prev_mask = pred_mask * 255
        
        mask_resized = transform.resize(prev_mask, img_shape)
        save_image(out_dir, sequence_name, frame_no, mask_resized)

def test_net(inp_dir,sequences=None,out_dir):
    if sequences == None:
        sequences=[name for name in os.listdir(inp_dir) if os.path.isdir(name)]
    
    net = build_test_model()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        net.initialize(sess, CHECKPOINT)
        
        for seq in sequences:
            test_sequence(sess, net, seq)
        
if __name__ == '__main__':
    pass