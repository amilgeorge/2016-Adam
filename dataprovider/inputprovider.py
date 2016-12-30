'''
Created on Dec 13, 2016

@author: george
'''
import numpy as np
from dataprovider.preprocess import vgg_preprocess

# class SequenceDB:
#     def __init__(self,sequence):
#         self.sequence = sequence
#         self.db=db
#         self.is_training = is_training
#         numImages = self.db.images.shape[0]
#         self.sequence_info = np.random.permutation(list(range(numImages)))
#         self.index = 0

class DB: pass

class DataBatch:pass
    

class DataIterator:
        
    def __init__(self,db, batch_size,is_training=True):
        self.db=db
        self.is_training = is_training
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
                
            if self.is_training:
                images = self.process_prev_mask_in_images(images)
                
                

            images = (images*255)
            images = vgg_preprocess(images)
                                
            # Prepare data batch
            batch = DataBatch()
            batch.images = images
            batch.labels = labels
            batch.weights = weights
            self.index += self.batch_size
            return batch
        else:       
            raise StopIteration()