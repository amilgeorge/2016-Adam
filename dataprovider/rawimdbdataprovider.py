# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:30:04 2016

@author: george
"""
import matplotlib
import os
import numpy as np
from skimage import io,transform,morphology
import re
from dataprovider.preprocess import vgg_preprocess, reverse_vgg_preprocess
from dataprovider import frame_no_calculator
from dataprovider.transformer_rand import ImageRandomTransformer
from numpy import dtype, size
import random
import skimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataprovider import imdb


import matplotlib.pyplot as plt
from dataprovider import inputhelper
from common import diskutils
import glob


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 854



    
class InputProvider:


    
    def __init__(self, db_name , prev_frame_calculator = None):
        
        self.prev_frame_calculator = prev_frame_calculator
        self.db = imdb.get_imdb(db_name)
        self.dataiter = None

           
            
    class DataBatch:pass
    
    class DataIterator:
        
        def __init__(self,db,prev_frame_calculator,batch_size,is_training):
            self.db = db
            self.prev_frame_calculator = prev_frame_calculator
            self.is_training = is_training
            self.img_size = [IMAGE_HEIGHT,IMAGE_WIDTH]
            self.img_channels = 7
            num_images = self.db.num_train_infos()
            print(num_images)
            self.sequence_info = np.random.permutation(list(range(num_images)))
            self.index = 0
            self.batch_size = batch_size
            self.transformer = ImageRandomTransformer(self._get_transform_config())
        
          
        def _get_transform_config(self):
            config ={}
            config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] =[-10,10]
            config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = None
            config[ImageRandomTransformer.CONFIG_SHEAR_RANGE] = [-5,5]
            config[ImageRandomTransformer.CONFIG_SHEAR_ANGLE_STEP] = None
            config[ImageRandomTransformer.CONFIG_SCALE_FACTOR_RANGE] = [1.0,1.5]
            config[ImageRandomTransformer.CONFIG_TRANSLATION_FACTOR_STEP] = 0.1
            config[ImageRandomTransformer.CONFIG_TRANSLATION_FACTOR_RANGE] = [-0.2,0.2]

            return config
        
        def __iter__(self):
            return self



        def __next__(self):
            to_index = self.index + self.batch_size
            if to_index <= len(self.sequence_info):


                selected_indexes = self.sequence_info[self.index:to_index]
                #print(selected_indexes)
                
                # Read images and labels 
                images = np.zeros([self.batch_size]+self.img_size+[self.img_channels],dtype = np.float32)
                labels = np.zeros([self.batch_size]+self.img_size)
                weights = np.empty([self.batch_size]+self.img_size,dtype=np.float32)

                for i,idx in enumerate(selected_indexes):
                    #import pdb
                    #pdb.set_trace()
                    image,label = self.db.get_at(idx, self.prev_frame_calculator)
                    label_dim = np.expand_dims(label,axis=2)

                    image = skimage.img_as_float(image)

                    stked = np.concatenate((image,label_dim),axis = 2)

                    stked,_ = self.transformer.get_random_transformed(stked)
                    stked = inputhelper.random_crop(stked,self.img_size)

                    image = stked[:,:,0:7]*255

                    if self.is_training:
                        image = self.process_prev_mask_in_image(image)

                    
                    label = np.uint8(stked[:,:,7])
                    ## Label as changes
                    images[i,:,:,:]=image
                    labels[i,:,:]=label

                    if ((np.count_nonzero(images[i,:, :, 3]) == 0)):
                        print("warning :imgprovider previmg all zeros")
                        labels[i, :, :] = 0

                    #weights[i,:,:] = inputhelper.get_distance_transform(label)

                    #weights[i,:,:] = inputhelper.get_weights_classwise_osvos_old(label)
                    #weights[i,:,:] = inputhelper.get_weights_osvos_distance(label)

                    #weights[i,:,:] = inputhelper.get_weights_classwise2(label,factor=3)
                    inputhelper.verify_input_img(images[i,:,:,:])


                # Prepare data batch
                batch = InputProvider.DataBatch()
                batch.images = images
                batch.labels = labels
                batch.weights = weights

                self.index += self.batch_size
                return batch
            else:       
                raise StopIteration()

        def next(self):
            return self.__next__()

        def _debug(self,img,label):
            plt.subplot(2,2,1)
            frame = plt.gca()
            frame.axes.set_title('Img')
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(np.uint8(img[:,:,0:3]*255))
            plt.subplot(2,2,2)
            frame = plt.gca()
            frame.axes.set_title('Prev Img')
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(np.uint8(img[:,:,4:7]*255))
            plt.subplot(2,2,3)
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(np.uint8(label))
            plt.subplot(2,2,4)
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(img[:,:,3])

        def get_data(self,selected_indexes):  
            if self.is_training:
                pass
            else:
                pass
                #image = transform.resize(image,resize)
                #return  
        
        def process_prev_mask_in_image(self,image):
            MASK_CHANNEL = 3
            prev_mask = image[:,:,MASK_CHANNEL]
            prev_mask_mod = self.add_noise_prev_mask(prev_mask)
            image[:,:,MASK_CHANNEL] = np.round(prev_mask_mod)
            return image
             
        def process_prev_mask_in_images(self,images):
            MASK_CHANNEL = 3
            for i in range(0,size(images,0)):
                prev_mask = images[i,:,:,MASK_CHANNEL]
                prev_mask_mod = self.add_noise_prev_mask(prev_mask)
                images[i,:,:,MASK_CHANNEL] = np.round(prev_mask_mod)  

            return images
                   
        def add_noise_prev_mask(self,prev_mask): 
            ERODE = 'Erode'
            DILATE = 'Dilate'
            NONE_ = 'None'
            options = [ERODE,DILATE,NONE_]
            #options = [DILATE,NONE_]
            #options = [NONE_]
            ch = random.choice(options)
            if ch == ERODE:
                sz = random.randint(2,3)
                new_mask = morphology.erosion(prev_mask,np.ones([sz,sz]))
                return new_mask
            elif ch == DILATE:

                sz = random.randint(2,4)
                new_mask = morphology.dilation(prev_mask,np.ones([sz,sz]))
                return new_mask
            else :
                return prev_mask
            
    def sequence_batch_itr(self, batch_size):
        return self.DataIterator(self.db,self.prev_frame_calculator, batch_size,is_training=True)

    def initialize_iterator(self,batch_size):
        self.batch_size = batch_size
        self.dataiter = self.DataIterator(self.db, self.prev_frame_calculator, batch_size, is_training=True)

    def next_mini_batch(self):
        try:
            minibatch = self.dataiter.next()
        except StopIteration:
            self.initialize_iterator(self.batch_size)
            minibatch = self.dataiter.next()

        return minibatch



def _debug(img,label,save_loc = None):
    plt.subplot(2,2,1)
    frame = plt.gca()
    frame.axes.set_title('Img')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(np.uint8(img[:,:,0:3]))
    plt.subplot(2,2,2)
    frame = plt.gca()
    frame.axes.set_title('Prev Img')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(np.uint8(img[:,:,4:7]))

    ax = plt.subplot(2,2,3)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    im = ax.imshow(np.uint8(label))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = plt.subplot(2,2,4)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    im = ax.imshow(img[:,:,3])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    if save_loc:
        plt.savefig(save_loc)

    plt.close()
   
def test_gen_samples():

    pfc_offset1 = frame_no_calculator.get(frame_no_calculator.POLICY_DEFAULT_ZERO,0)
    db_name = imdb.IMDB_DAVIS_COMBO
    provider = InputProvider(db_name=db_name,prev_frame_calculator=pfc_offset1)
    num=0
    dir = 'davis_imdb_samp_{}_off0/'.format(db_name)
    diskutils.ensure_dir(dir)
    max_num = 1000
    batch_size = 4
    while num < max_num:
        input_batch = provider.sequence_batch_itr(batch_size)

        for i, batch in enumerate(input_batch):
            print (i, 'rgb files: ')

            if num > max_num:
                break

            imgs_branch1 = reverse_vgg_preprocess(batch.images)

            for j in range(batch_size):
                if num > max_num:
                    break
                save_loc = os.path.join(dir, '{}.png'.format(num))

                _debug(imgs_branch1[j,:,:,:],batch.labels[j,:,:],save_loc)
                num+=1

if __name__ == '__main__':
    provider = InputProvider(resize=[IMAGE_HEIGHT,IMAGE_WIDTH], )
    batch_size = 4
    num=0
    dir = '/tmp/davis_samp_1/'
    diskutils.ensure_dir(dir)
    max_num = 1000
    while num < max_num:
        input_batch = provider.sequence_batch_itr(batch_size)

        for i, batch in enumerate(input_batch):

            if num > max_num:
                break

            for j in range(batch_size):
                if num > max_num:
                    break
                save_loc = os.path.join(dir, '{}.png'.format(num))

                imgs= reverse_vgg_preprocess(batch.images)
                _debug(imgs[j,:,:,:],batch.labels[j,:,:],save_loc)
                print (i, 'rgb files: ')
                num+=1

    
    print("Done")
