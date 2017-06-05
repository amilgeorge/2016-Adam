'''
Created on Dec 1, 2016

@author: george
'''
# -*- coding: utf-8 -*-
from net.segnet2 import NUM_CLASSES
import tensorflow as tf
import numpy as np
from dataprovider.imdbdataprovider import InputProvider
from dataprovider import frame_no_calculator as fnc

from common.logger import getLogger
from common.diskutils import ensure_dir
from net import segnet2 as segnet
import time
import os
import test_segnet2
from dataprovider import imdb
#import tracemalloc
#import gc

slim = tf.contrib.slim


OFFSETS = [(fnc.POLICY_OFFSET,1)]
#OFFSETS = [(fnc.POLICY_DEFAULT_ZERO,0)]
offset_string = "-".join(str(x) for p,x in OFFSETS)
dbname = imdb.IMDB_CUSTOM_MASK_DAVIS2016
lr = 1e-2

RUN_ID = "s480p-{}-O{}-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-3".format(dbname,offset_string)

START_CHECKPOINT = "exp_repo/s480pvgg-daviscombo-O1-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-3/iters-50000"
START_MASK_FOLDER = "test_out/s480pvgg-daviscombo-O1-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-3/iter-50000/480p"

EVENTS_DIR = os.path.join('events',RUN_ID)#time.strftime("%Y%m%d-%H%M%S")
EXP_DIR = os.path.join('exp',RUN_ID)
LOGS_DIR = os.path.join('logs',RUN_ID)

IMG_HEIGHT = 480
IMG_WIDTH = 854


    
def train():

    SAVE_STEP_SIZE = 3000
    custom_mask_folder = START_MASK_FOLDER

    ensure_dir(EXP_DIR)
    ensure_dir(LOGS_DIR)        
    logger = getLogger(os.path.join(LOGS_DIR,time.strftime("%Y%m%d-%H%M%S")+'.log'))

    RNG_SEED = 3
    np.random.seed(RNG_SEED)
    pfc = fnc.get(OFFSETS[0][0],OFFSETS[0][1])

    with tf.Graph().as_default():
        tf.set_random_seed(1)

        NUM_CLASSES = segnet.NUM_CLASSES
        # Create placeholders for input and output
        inp = tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,7],name='input')
        label =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH],name='label')
        weights =  tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH],name='weights')
        keep_prob = tf.placeholder(tf.float32)

        is_training_pl = tf.placeholder(tf.bool,name="segnet_is_training")


                    
        #loss,logit = segnet.inference(inp, label, is_training_pl,weights)
        #loss,logit = segnet.inference_vgg16(inp, label, is_training_pl,weights)
        loss,logit = segnet.inference_vgg16_withskip(inp, label, is_training_pl,weights)
        #loss,logit = segnet.inference_vgg16_withdrop(inp, label, is_training_pl,weights,keep_prob)


        loss_averages_op = segnet._add_loss_summaries(loss)
        #segnet.prepare_encoder_parameters()


        logit = tf.reshape(logit, (-1, NUM_CLASSES))
        out=tf.reshape(tf.nn.softmax(logit),[-1,IMG_HEIGHT,IMG_WIDTH,2])

        net_dict = {"inp_pl": inp,
                   "label_pl": label,
                   "out": out,
                   "is_training_pl": is_training_pl,
                   "loss": loss}

        #out = tf.nn.softmax(logit)
        predictions = tf.arg_max(out, 3, "predictions")

        # Declare the optimizer
        global_step_var = tf.Variable(1, trainable=False)
            
        #learning_rate = tf.train.exponential_decay(0.01, global_step_var, 10,
        #                               0.1, staircase=True)
        #boundaries = [100000, 200000]
        #values = [1e-2, 1e-3, 1e-4]
        #learning_rate = tf.train.piecewise_constant(global_step_var, boundaries, values)

        with tf.control_dependencies([loss_averages_op]):
        #optimizer = tf.train.GradientDescentOptimizer(0.01)
            optimizer = tf.train.MomentumOptimizer(lr,0.9)
            #logger.info("using lr: {}".format(lr))
            #optimizer = tf.train.AdamOptimizer(lr)

            gradients = optimizer.compute_gradients(loss)
            
        
                    
            
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step_var)
        #apply_gradient_op = segnet.train(loss, global_step_var)    
        max_iters = 500000
        batch_size = segnet.BATCH_SIZE
            
        # Input Provider
        #inputProvider = SampleInputProvider(resize=[IMG_HEIGHT,IMG_WIDTH],is_dummy=False)
        inputProvider = InputProvider(db_name = dbname,prev_frame_calculator=pfc)
            
    
        ##### Summaries #######
        prediction_reshaped = tf.reshape(predictions, [-1])
        label_reshaped=tf.reshape(tf.cast(label, tf.int64), [-1])

        acc = tf.contrib.metrics.accuracy(prediction_reshaped, label_reshaped)
        confusion_matrix = tf.contrib.metrics.confusion_matrix(prediction_reshaped,label_reshaped,num_classes=2)

        #regularization_loss = tf.contrib.losses.get_regularization_losses()

        tp_tensor = confusion_matrix[1,1]
        fp_tensor = confusion_matrix[1,0]
        fn_tensor = confusion_matrix[0,1]

        precision = tp_tensor/(tp_tensor + fp_tensor)
        recall = tp_tensor/(tp_tensor + fn_tensor)
        jaccard = tp_tensor/(tp_tensor + fn_tensor + fp_tensor)

        out_reshaped = tf.reshape(out,[-1,IMG_HEIGHT,IMG_WIDTH,NUM_CLASSES])
        tf.summary.image('/output',tf.expand_dims(out_reshaped[:,:,:,1],3))
        tf.summary.image('/label',tf.expand_dims(label,3))

        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
                
        tf.summary.scalar('/loss', loss)
        tf.summary.scalar('/accuracy',acc)
        tf.summary.scalar('/precision',precision)
        tf.summary.scalar('/recall',recall)
        tf.summary.scalar('/jaccard',jaccard)
        #tf.scalar_summary('/regularization_loss', regularization_loss)

        merged_summary = tf.summary.merge_all()
        
        VALIDATION_SUMMARIES = 'validation_summaries'
        
        val_loss_pl = tf.placeholder(tf.float32)
        val_acc_pl = tf.placeholder(tf.float32)
        val_precision_pl = tf.placeholder(tf.float32)
        val_recall_pl = tf.placeholder(tf.float32)
        val_jaccard_pl = tf.placeholder(tf.float32)


        val_loss_summary = tf.summary.scalar('/val/loss', val_loss_pl,collections=VALIDATION_SUMMARIES)
        val_acc_summary = tf.summary.scalar('/val/accuracy', val_acc_pl,collections=VALIDATION_SUMMARIES)
        val_precision_summary = tf.summary.scalar('/val/precision', val_precision_pl,collections=VALIDATION_SUMMARIES)
        val_recall_summary = tf.summary.scalar('/val/recall', val_recall_pl ,collections=VALIDATION_SUMMARIES)
        val_jaccard_summary = tf.summary.scalar('/val/jaccard', val_jaccard_pl ,collections=VALIDATION_SUMMARIES)


        merged_val_summary = tf.summary.merge([val_loss_summary,val_acc_summary,val_precision_summary,val_recall_summary,val_jaccard_summary ],
                                              collections=None)
        ########################
        
        saver = tf.train.Saver(max_to_keep = 10)
        #saver.export_meta_graph("metagraph.meta", as_text=True)


        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

            # Use predefined checkpoint
            if START_CHECKPOINT is None:
                checkpoint_file = tf.train.latest_checkpoint(EXP_DIR);

            else:
                checkpoint_file = START_CHECKPOINT

            if checkpoint_file:
                logger.info('using checkpoint :{}'.format(checkpoint_file))

                saver.restore(sess, checkpoint_file)

            else :   
                # Build an initialization operation to run below.
                logger.info('initializing network from pretrained weights')
                init = tf.global_variables_initializer()

                # Start running operations on the Graph.
                sess.run(init)

                segnet.initialize_vgg16(sess)

            train_summary_writer = tf.summary.FileWriter(EVENTS_DIR + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(EVENTS_DIR + '/test')

            logger.info('initializing db policy')
            inputProvider.db.set_policy(imdb.CustomMaskDB2016.POLICY_SELECT_CM, 0.85)
            logger.info('setting custom mask folder:{}'.format(custom_mask_folder))
            inputProvider.db.set_mask_folder(custom_mask_folder)
            logger.info('initializing iterator')
            inputProvider.initialize_iterator(batch_size)
            #logger.info('initializing fetcher')
            #inputProvider.intitialize_fetcher()

            while global_step_var.eval() <= max_iters:
                step = global_step_var.eval()
                sequence_batch = inputProvider.next_mini_batch_sync()


                result = sess.run([apply_gradient_op, loss,merged_summary,acc,precision,recall,jaccard,tp_tensor,
                                   fn_tensor,fp_tensor,confusion_matrix],
                                  feed_dict={inp:sequence_batch.images,
                                            label:sequence_batch.labels,
                                            weights:sequence_batch.weights,
                                            is_training_pl:True})#,
                                            #keep_prob: 0.5})
                loss_value = result[1]
                acc_value = result[3]
                precision_value = result[4]
                recall_value = result[5]
                jaccard_value = result[6]

                logger.info('iters:{}, loss :{} accuracy:{} p:{} r:{} j:{}'.format(step, loss_value,acc_value,precision_value,recall_value,jaccard_value))

                if step%100 ==0:
                    #import pdb
                    #pdb.set_trace()
                    train_summary_writer.add_summary(result[2], step )


                #io.imshow(out.eval())
                #pass


                if step % SAVE_STEP_SIZE == 0:
                    logger.info('Saving weights.')
                    saver.save(sess, os.path.join(EXP_DIR,'iters'),global_step = step)
                    logger.info('Flushing .')
                    train_summary_writer.flush()
                    test_summary_writer.flush()

                if step % SAVE_STEP_SIZE == 0:
                    test_out_dir = os.path.join('test_out',RUN_ID,'iter-{}'.format(step))
                    ensure_dir(test_out_dir)
                    test_segnet2.test_network(sess,net_dict,test_out_dir,pfc)
                    mask_folder = os.path.join(test_out_dir, '480p')
                    logger.info('setting custom mask folder:{}'.format(mask_folder))
                    inputProvider.db.set_mask_folder(mask_folder)
                
            train_summary_writer.close()
            test_summary_writer.close()
    
    
if __name__ == '__main__':
    train()
