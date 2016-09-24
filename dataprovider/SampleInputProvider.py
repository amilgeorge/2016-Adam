# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:30:04 2016

@author: george
"""

import os
import numpy as np
from skimage import io,transform
import matplotlib.pyplot as plt
import re
from dataprovider.preprocess import vgg_preprocess

class SampleInputProvider:
    
    BASE_DIR = os.path.join(os.path.expanduser('~'),'DAVIS')
    IMAGESETS = os.path.join('ImageSets','480p')
    
    RESIZE_HEIGHT = 224
    RESIZE_WIDTH = 224
    
    def __init__(self):
        self.trainsetInfo = np.loadtxt(os.path.join(self.BASE_DIR,\
                        self.IMAGESETS,'train.txt'), dtype=bytes,unpack=False).astype(str)
        
        self.trainsetInfo = self.trainsetInfo[1:2,:]
        self.db = self.createDB()
    
    class DB: 
        def __init__(self,):
            pass
           
            
    class DataBatch:pass
    
    class DataIterator:
        
        def __init__(self,db, batch_size):
            self.db=db
            numImages = self.db.images.shape[0]
            self.sequence_info = list(range(numImages))
            self.index = 0
            self.batch_size = batch_size
          
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.index < len(self.sequence_info):
                
                
                toIndex = self.index+self.batch_size
                
                # Read images and labels 
                images = self.db.images[self.index:toIndex,:,:,:]
                images = (images*255)
                images = vgg_preprocess(images)
                
                #rgb_files = [rgb_file - mean for rgb_file in rgb_files]
                
                labels = self.db.labels[self.index:toIndex,:,:]
                
                # Prepare data batch
                batch = SampleInputProvider.DataBatch()
                batch.images = images
                batch.labels = labels
                
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
        
        trainSetInfo = self.trainsetInfo
        numImages = trainSetInfo.shape[0]
        db = SampleInputProvider.DB()
        db.images = np.zeros([numImages,self.RESIZE_HEIGHT,self.RESIZE_WIDTH,4])
        db.labels = np.zeros([numImages,self.RESIZE_HEIGHT,self.RESIZE_WIDTH])
        db.filenames = []
        
        for i in range(numImages):
            imageFile = trainSetInfo[i,0]
            labelFile = trainSetInfo[i,1]
            prevMaskFile = self.getPrevMaskFile(labelFile)
            image = self.readImage(imageFile,prevMaskFile)
            label = self.readLabel(labelFile)
            db.images[i,:,:,:] = image
            db.labels[i,:,:] = label
            db.filenames.append(imageFile)
        return db
            
    def readImage(self,imageFile,prevMaskFile):
        
        # Fix full file path
        rgbFile =self.BASE_DIR + imageFile
        maskFile = self.BASE_DIR + prevMaskFile
        
        # Read images 
        rgb = io.imread(rgbFile)
        mask = io.imread(maskFile)
        mask = np.expand_dims(mask,axis=2)
        
        # Concatenate images
        image =  np.concatenate((rgb, mask), 2)
        
        # Resize 
        image = transform.resize(image,[self.RESIZE_HEIGHT,self.RESIZE_WIDTH])

        #io.imshow(image[:,:,0:3])        
        
        return image
    
    def readLabel(self,labelFile) :
        maskFile = self.BASE_DIR + labelFile
        mask = io.imread(maskFile)
        
        # Resize 
        mask = transform.resize(mask,[self.RESIZE_HEIGHT,self.RESIZE_WIDTH])
        
        #io.imshow(mask)
        
        return mask
    
    def read_rgb_image(filepath):
        rgb_img = ndimage.imread(filepath)
        width = height = 224
        img_width = rgb_img.shape[1]
        img_height = rgb_img.shape[0]
    
        # scale such that smaller dimension is 256
        if img_width < img_height:
            factor = 256.0 / img_width
        else:
            factor = 256.0 / img_height
        rgb_img = transform.rescale(rgb_img, factor, preserve_range=True)
    
        # crop randomly
        width_start = np.random.randint(0, rgb_img.shape[1] - width)
        height_start = np.random.randint(0, rgb_img.shape[0] - height)
    
        rgb_img = rgb_img[height_start:height_start + height, width_start:width_start + width]
        return rgb_img

if __name__ == '__main__':
    provider = SampleInputProvider()
    input_batch = provider.sequence_batch_itr(1)
    for i, batch in enumerate(input_batch):
        print (i, 'rgb files: ')