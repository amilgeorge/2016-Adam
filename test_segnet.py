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




CHECKPOINT = CHECKPOINT_17
davis = DataAccessHelper()
logger = getLogger() 

RUN_ID = "f-segnet-ch7-aug-10"
EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

def threshold_image(prediction,threshold=0.5):
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    return prediction

def save_image(out_dir,sequence_name,frame_no,img):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    io.imsave(out_path, img)

def build_model_for_fine_tuning():
    inp = tf.placeholder(tf.float32,shape=[None,224,224,7],name='input')
    label =  tf.placeholder(tf.float32,shape=[None,224,224],name='label')
    is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")
    loss,logit = segnet.inference(inp, label, is_training_pl)

    logit = tf.reshape(logit, (-1, segnet.NUM_CLASSES))
    out = tf.reshape(tf.nn.softmax(logit),[-1,224,224,segnet.NUM_CLASSES])
    
    ret_val = {"inp_pl":inp,
               "label_pl":label,
               "out" : out,
               "is_training_pl":is_training_pl,
               "loss":loss }
    return ret_val
    
def build_test_model():
    inp = tf.placeholder(tf.float32,shape=[None,224,224,7],name='input')
    is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")
    logit = segnet.inference(inp,labels=None, phase_train=is_training_pl)      
    
    #out=tf.reshape(tf.nn.softmax(logit),[-1,224,224,2])
    logit = tf.reshape(logit, (-1, segnet.NUM_CLASSES))
    out = tf.reshape(tf.nn.softmax(logit),[-1,224,224,segnet.NUM_CLASSES])
    ret_val = {"inp_pl":inp,
               "out" : out,
               "is_training_pl":is_training_pl }
    return ret_val
    
def test_sequence(session,net,sequence_name,out_dir,keep_size = True):
    
    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path, [224,224])*255
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
        result = session.run([out], 
                             feed_dict={inp:inp_img,
                             is_training_pl:False})
        prediction = result[0][0,:,:,1]

        #print(result[0][0,1,1,1],result[0][0,1,1,0])
        
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
        restorer2 = tf.train.Saver()
        restorer2.restore(sess,CHECKPOINT)
        
        for seq in sequences:
            logger.info('Testing sequence: {}'.format(seq))
            test_sequence(sess, net, seq,out_dir)

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
    summary_writer = tf.train.SummaryWriter(events_dir, session.graph)
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
    tf.scalar_summary('/train/loss', loss)  
    val_loss_pl = tf.placeholder(tf.float32)
    VALIDATION_SUMMARIES = 'validation_summaries'
    val_loss_summary = tf.scalar_summary('/val/loss', val_loss_pl,collections=VALIDATION_SUMMARIES)  
    merged_val_summary = tf.merge_summary([val_loss_summary],collections=None)   
    merged_summary = tf.merge_all_summaries()    
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
    test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
    #test_sequences = ['paragliding-launch']
    m=re.match(r"exp/(.*)/iter.*",CHECKPOINT)
    res_dir = m.group(1)+"-1"
    out_dir = "../Results/{}/480p".format(res_dir)
    logger.info("Output to: {}".format(out_dir))    
    test_net(test_sequences, out_dir=out_dir)
