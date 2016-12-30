# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:30:04 2016

@author: george
"""
import matplotlib
#matplotlib.use('Agg')
import os
import numpy as np
from skimage import io,transform,morphology
import re
from dataprovider.preprocess import vgg_preprocess
from dataprovider.davis import DataAccessHelper
from dataprovider.transformer import ImageRandomTransformer
from numpy import dtype, size
import random

import matplotlib.pyplot as plt
from binstar_client.utils.notebook.data_uri import Image
from dataprovider import inputhelper

davis = DataAccessHelper()
def prepare_input_ch7(image_path,prev_mask):
    
    # Read image
    img = davis.read_image(image_path, [224,224])
    img = img*255
    
    mask = np.expand_dims(prev_mask,axis=2)    
    
    # Read previous image
    prev_img_path = davis.construct_image_path(image_path, offset= -1)
    prev_img = davis.read_image(prev_img_path, [224,224])
    prev_img = prev_img*255
    
    # Concatenate images
    inp_img =  np.concatenate((img, mask,prev_img), 2)
    
    # Apend axis for batch size
    inp_img = np.expand_dims(inp_img,axis=0)    
    
    
    inp_img = vgg_preprocess(inp_img)
    
    return inp_img
   
def prepare_input(image_path,prev_mask):
    
    # Read image
    img = davis.read_image(image_path, [224,224])
    img = img*255
    
    mask = np.expand_dims(prev_mask,axis=2)    
    
    # Concatenate images
    inp_img =  np.concatenate((img, mask), 2)
    
    # Apend axis for batch size
    inp_img = np.expand_dims(inp_img,axis=0)    
    
    
    inp_img = vgg_preprocess(inp_img)
    
    return inp_img
    
    
class SampleInputProvider:
    
    BASE_DIR = os.path.join('/work/george','DAVIS')
    IMAGESETS = os.path.join('ImageSets','480p')
    
    RESIZE_HEIGHT = 224
    RESIZE_WIDTH = 224
    NUM_CHANNELS = 7
    
    COARSE_OUT_HEIGHT = 56
    COARSE_OUT_WIDTH = 56
    
    def __init__(self,is_coarse=True,is_dummy = False):
        
        self.davis = DataAccessHelper()
        self.resize = [SampleInputProvider.RESIZE_HEIGHT,SampleInputProvider.RESIZE_WIDTH]
        if is_coarse:
            self.weights_resize = [SampleInputProvider.COARSE_OUT_HEIGHT,SampleInputProvider.COARSE_OUT_WIDTH]
        else:
            self.weights_resize = [SampleInputProvider.RESIZE_HEIGHT,SampleInputProvider.RESIZE_WIDTH]
        
        self.train_set_info = np.loadtxt(os.path.join(self.BASE_DIR,\
                        self.IMAGESETS,'train.txt'), dtype=bytes,unpack=False).astype(str)
        self.val_set_info = np.loadtxt(os.path.join(self.BASE_DIR,\
                        self.IMAGESETS,'val.txt'), dtype=bytes,unpack=False).astype(str)
        #self.val_set_info = self.val_set_info[:419,:]
        if is_dummy:
            self.train_set_info = self.train_set_info[247:250,:]
            self.val_set_info = self.val_set_info[247:250,:]
        
        self.db = self.createDB(self.train_set_info)
        self.validation_db = self.createDB(self.val_set_info)
    
    class DB: 
        def __init__(self,):
            pass
           
            
    class DataBatch:pass
    
    class DataIterator:
        
        def __init__(self,db, batch_size,is_training=True):
            self.db=db
            self.is_training = is_training
            numImages = self.db.images.shape[0]
            self.img_size = list(self.db.images.shape[1:3])
            self.weights_size = list(self.db.weights.shape[1:3])
            self.img_channels = self.db.images.shape[3]
            self.sequence_info = np.random.permutation(list(range(numImages)))
            self.index = 0
            self.batch_size = batch_size
            self.transformer = ImageRandomTransformer(self._get_transform_config())
        
          
        def _get_transform_config(self):
            config ={}
            config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] =[-10,10]
            config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = 5
            config[ImageRandomTransformer.CONFIG_SHEAR_RANGE] = [-5,5]
            config[ImageRandomTransformer.CONFIG_SHEAR_ANGLE_STEP] = None
            config[ImageRandomTransformer.CONFIG_SCALE_FACTOR_RANGE] = [0.7,1.3]
            config[ImageRandomTransformer.CONFIG_TRANSLATION_FACTOR_STEP] = 0.1
            config[ImageRandomTransformer.CONFIG_TRANSLATION_FACTOR_RANGE] = [-0.2,0.2]

            return config
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.index < len(self.sequence_info):
                
                
                toIndex = self.index+self.batch_size
                
                selected_indexes = self.sequence_info[self.index:toIndex]
                
                # Read images and labels 
                images = np.zeros([self.batch_size]+self.img_size+[self.img_channels],dtype = np.float32)
                labels = np.zeros([self.batch_size]+self.img_size)
                weights = np.empty([self.batch_size]+self.weights_size,dtype=np.float32)
                for i,idx in enumerate(selected_indexes):
                    label_dim = np.expand_dims(self.db.labels[idx,:,:], axis=2)
                    stked = np.concatenate((self.db.images[idx,:,:,:],label_dim),axis = 2)
                    
                    stked,_ = self.transformer.get_random_transformed(stked)
                    image = stked[:,:,0:7]
                    if self.is_training:
                        image = self.process_prev_mask_in_image(image)
                    
                    label = np.uint8(stked[:,:,7])    
                    images[i,:,:,:]=image
                    labels[i,:,:]=label
                    weights[i,:,:] = inputhelper.get_weights_classwise2(label,resize=self.weights_size,factor=3)
                    #plt.figure()
                    #plt.imshow(weights[i,:,:])
                    #self._debug(image,label)
                
                #images = self.db.images[selected_indexes,:,:,:]
                #labels = self.db.labels[selected_indexes,:,:]
                #weights = self.db.weights[selected_indexes,:,:]
                
               
                
               

                images = (images*255)
                images = vgg_preprocess(images)
                                
                # Prepare data batch
                batch = SampleInputProvider.DataBatch()
                batch.images = images
                batch.labels = labels
                batch.weights = weights
                self.index += self.batch_size
                return batch
            else:       
                raise StopIteration()
        
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
                #plt.figure()
                #plt.subplot(1,2,1)
                #plt.imshow(prev_mask)
            prev_mask_mod = self.add_noise_prev_mask(prev_mask)
            image[:,:,MASK_CHANNEL] = np.round(prev_mask_mod)  
           
                #plt.subplot(1,2,2)
                #plt.imshow(prev_mask_mod)  
                #plt.show()
            return image
             
        def process_prev_mask_in_images(self,images):
            MASK_CHANNEL = 3
            for i in range(0,size(images,0)):
                prev_mask = images[i,:,:,MASK_CHANNEL]
                #plt.figure()
                #plt.subplot(1,2,1)
                #plt.imshow(prev_mask)
                prev_mask_mod = self.add_noise_prev_mask(prev_mask)
                images[i,:,:,MASK_CHANNEL] = np.round(prev_mask_mod)  
           
                #plt.subplot(1,2,2)
                #plt.imshow(prev_mask_mod)  
                #plt.show()
            return images
                   
        def add_noise_prev_mask(self,prev_mask): 
            ERODE = 'Erode'
            DILATE = 'Dilate'
            NONE_ = 'None'
            options = [ERODE,DILATE,NONE_]
            #options = [NONE_]
            ch = random.choice(options)
            if ch == ERODE:
                sz = random.randint(2,4)
                new_mask = morphology.erosion(prev_mask,np.ones([sz,sz]))
                return new_mask
            elif ch == DILATE:

                sz = random.randint(2,4)
                new_mask = morphology.dilation(prev_mask,np.ones([sz,sz]))
                return new_mask
            else :
                return prev_mask
            
    def sequence_batch_itr(self, batch_size):
        return self.DataIterator(self.db, batch_size)
    
    def val_seq_batch_itr(self, batch_size):
        return self.DataIterator(self.validation_db, batch_size,is_training=False)
    
    def getPrevMaskFile(self,labelFile):
        m=re.match(r"(/.*/.*/.*/)(.*).png",labelFile)
        prefix = m.group(1)
        frameNo = int(m.group(2))
        maskFrameNo = frameNo-1 if frameNo -1 >=0 else 0
        
        
        prevMaskFile='{0}{1:05}.png'.format(prefix,maskFrameNo)
        return prevMaskFile
    
        
    def createDB(self,data_set_info):
        #import cPickle as pickle
        cached_db = '/work/george/cache/davisdb.pickle'
        if os.path.isfile(cached_db):
        #    with open(cached_db,'rb') as f:  
        #        db = 
        #    return db
            pass
        else:
            numImages = data_set_info.shape[0]
            db = SampleInputProvider.DB()
            db.images = np.zeros([numImages,self.RESIZE_HEIGHT,self.RESIZE_WIDTH,SampleInputProvider.NUM_CHANNELS])
            db.labels = np.zeros([numImages,self.RESIZE_HEIGHT,self.RESIZE_WIDTH])
            db.weights = np.zeros([numImages]+self.weights_resize,dtype=np.float32)
            db.filenames = []
            
            for i in range(numImages):
                imageFile = data_set_info[i,0]
                imageFile = imageFile[1:]
                labelFile = data_set_info[i,1]
                labelFile = labelFile[1:]
                #prevMaskFile = self.getPrevMaskFile(labelFile)
                image = self.read_image2(imageFile)
                label = self.davis.read_label(labelFile, self.resize)
                db.images[i,:,:,:] = image
                db.labels[i,:,:] = label
                db.weights[i,:,:] = self.davis.get_weight_map(imageFile, resize = self.weights_resize)
                db.filenames.append(imageFile)
            
            #with open(cached_db,'wb') as f:  
            #    pickle.dump(db,f)
            
            return db
    
    def read_image2(self,image_path,prev_mask = None):  
    
        
        prev_rgb_path = self.davis.construct_image_path(image_path, offset= -1)
        
        rgb = self.davis.read_image(image_path, self.resize)
        
        if prev_mask ==None:
            prev_mask_path = self.davis.construct_label_path(image_path, offset = -1)
            prev_mask = self.davis.read_label(prev_mask_path, self.resize)
            
        prev_mask = np.expand_dims(prev_mask, axis=2)
        
        prev_rgb = self.davis.read_image(prev_rgb_path, self.resize)
        
        # Concatenate 
        image =  np.concatenate((rgb, prev_mask,prev_rgb), 2)
        
        return image
              
    def readImage(self,imageFile,prevMaskFile):
        
        # Fix full file path
        rgbFile =self.BASE_DIR + imageFile
        maskFile = self.BASE_DIR + prevMaskFile
        
        # Read images 
        rgb = io.imread(rgbFile)
        mask = io.imread(maskFile,as_grey=True)
        mask = np.expand_dims(mask,axis=2)
        
        # Concatenate images
        image =  np.concatenate((rgb, mask), 2)
        
        # Resize 
        image = transform.resize(image,[self.RESIZE_HEIGHT,self.RESIZE_WIDTH])

        #io.imshow(image[:,:,0:3])        
        
        return image
    
    def readLabel(self,labelFile) :
        maskFile = self.BASE_DIR + labelFile
        mask = io.imread(maskFile,as_grey=True)
        
        # Resize 
        mask = transform.resize(mask,[self.RESIZE_HEIGHT,self.RESIZE_WIDTH])
        
        #io.imshow(mask)
        
        return mask
    
   
    
if __name__ == '__main__':
    provider = SampleInputProvider(is_coarse=False,is_dummy=True)
    for j in range(1,1000):
        input_batch = provider.sequence_batch_itr(16)

        for i, batch in enumerate(input_batch):
            print (i, 'rgb files: ')
    
    print("Done")
