# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:30:04 2016

@author: george
"""

import os
import numpy as np
from skimage import io,transform
import re
from dataprovider.preprocess import vgg_preprocess
from dataprovider.davis import DataAccessHelper
from numpy import dtype
   
def prepare_input(img,prev_mask):
    
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
    
    COARSE_OUT_HEIGHT = 56
    COARSE_OUT_WIDTH = 56
    
    def __init__(self,is_coarse=True,is_dummy = False):
        
        self.data_helper = DataAccessHelper()
        self.resize = [SampleInputProvider.RESIZE_HEIGHT,SampleInputProvider.RESIZE_WIDTH]
        if is_coarse:
            self.weights_resize = [SampleInputProvider.COARSE_OUT_HEIGHT,SampleInputProvider.COARSE_OUT_WIDTH]
        else:
            self.weights_resize = [SampleInputProvider.RESIZE_HEIGHT,SampleInputProvider.RESIZE_WIDTH]
        
        self.trainsetInfo = np.loadtxt(os.path.join(self.BASE_DIR,\
                        self.IMAGESETS,'train.txt'), dtype=bytes,unpack=False).astype(str)
        
        if is_dummy:
            self.trainsetInfo = self.trainsetInfo[247:250,:]
        
        self.db = self.createDB()
    
    class DB: 
        def __init__(self,):
            pass
           
            
    class DataBatch:pass
    
    class DataIterator:
        
        def __init__(self,db, batch_size):
            self.db=db
            numImages = self.db.images.shape[0]
            self.sequence_info = np.random.permutation(list(range(numImages)))
            self.index = 0
            self.batch_size = batch_size
          
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.index < len(self.sequence_info):
                
                
                toIndex = self.index+self.batch_size
                
                selected_indexes = self.sequence_info[self.index:toIndex]
                
                # Read images and labels 
                images = self.db.images[selected_indexes,:,:,:]
                labels = self.db.labels[selected_indexes,:,:]
                weights = self.db.weights[selected_indexes,:,:]

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
    
    def sequence_batch_itr(self, batch_size):
        return self.DataIterator(self.db, batch_size)

    def getPrevMaskFile(self,labelFile):
        m=re.match(r"(/.*/.*/.*/)(.*).png",labelFile)
        prefix = m.group(1)
        frameNo = int(m.group(2))
        maskFrameNo = frameNo-1 if frameNo -1 >=0 else 0
        
        
        prevMaskFile='{0}{1:05}.png'.format(prefix,maskFrameNo)
        return prevMaskFile
        
    def createDB(self):
        #import cPickle as pickle
        cached_db = '/work/george/cache/davisdb.pickle'
        if os.path.isfile(cached_db):
        #    with open(cached_db,'rb') as f:  
        #        db = 
        #    return db
            pass
        else:
            trainSetInfo = self.trainsetInfo
            numImages = trainSetInfo.shape[0]
            db = SampleInputProvider.DB()
            db.images = np.zeros([numImages,self.RESIZE_HEIGHT,self.RESIZE_WIDTH,4])
            db.labels = np.zeros([numImages,self.RESIZE_HEIGHT,self.RESIZE_WIDTH])
            db.weights = np.zeros([numImages]+self.weights_resize,dtype=np.float32)
            db.filenames = []
            
            for i in range(numImages):
                imageFile = trainSetInfo[i,0]
                labelFile = trainSetInfo[i,1]
                prevMaskFile = self.getPrevMaskFile(labelFile)
                image = self.readImage(imageFile,prevMaskFile)
                label = self.readLabel(labelFile)
                db.images[i,:,:,:] = image
                db.labels[i,:,:] = label
                db.weights[i,:,:] = self.data_helper.get_weight_map(imageFile, resize = self.weights_resize)
                db.filenames.append(imageFile)
            
            #with open(cached_db,'wb') as f:  
            #    pickle.dump(db,f)
            
            return db
            
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
    provider = SampleInputProvider()
    input_batch = provider.sequence_batch_itr(2)
    for i, batch in enumerate(input_batch):
        print (i, 'rgb files: ')