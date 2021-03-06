'''
Created on Oct 7, 2016

@author: george
'''

import os
from skimage import io,transform,morphology
import numpy as np
import re
import yaml
from PIL import Image
import skimage

class DataAccessHelper(object):
    '''
    Helper class that provides convenient methods to access data from 
    davis data set
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.base_dir = os.path.join('/work/george','DAVIS/')
        
        self.image_path_prefix = os.path.join('JPEGImages','480p')
        self.annotations_prefix = os.path.join('Annotations','480p')
        self.db_info = os.path.join('Annotations','db_info.yml')
    def read_db_info(self):
        with open(self.__fullpath(self.db_info),'r') as f:
            db_info = yaml.load(f)
            
            return db_info
    
    def train_sequence_list(self):
        db_info = self.read_db_info()
        db_sequences = db_info['sequences']
        test_seqs = filter(lambda s: 'training' == s['set'] ,db_sequences)
        
        test_seq_names = [seq['name'] for seq in test_seqs]
        return test_seq_names
    
    def test_sequence_list(self):
        db_info = self.read_db_info()
        db_sequences = db_info['sequences']
        test_seqs = filter(lambda s: 'test' == s['set'] ,db_sequences)
        
        test_seq_names = [seq['name'] for seq in test_seqs]
        return test_seq_names
    
        
    def image_path(self,sequence_name,frame_no):
        image_path = os.path.join(self.image_path_prefix,sequence_name,'{0:05}.jpg'.format(frame_no))
        return image_path
    
    def label_path(self,sequence_name,frame_no):
        label_path = os.path.join(self.annotations_prefix,sequence_name,'{0:05}.png'.format(frame_no))
        return label_path
    
    def all_frames_nums(self,sequence_name):
        sequence_dir = self.__fullpath(os.path.join(self.image_path_prefix,sequence_name))
        
        img_collection = io.ImageCollection(sequence_dir+"/*.jpg")
	
        frames = map(lambda fn: int(os.path.splitext(os.path.basename(fn))[0]),img_collection.files)
        return list(frames)
    
    def split_path(self,path):
        

        m=re.match(r"(.*)/(.*/)(.*)/(.*).jpg",path)
        
        sequence_name = m.group(3)
        frame_no = int(m.group(4))
        return sequence_name,frame_no
    
    def __fullpath(self,image_path): 
        
        return os.path.join(self.base_dir, image_path)
    
    def image_shape(self,image_path):
        img = io.imread(self.__fullpath(image_path))  
        return img.shape
        
    def read_image(self, image_path, resize = None):
        
        image = io.imread(self.__fullpath(image_path))
        
        if resize != None:
            image = transform.resize(image,resize)
        
        return image

    def threshold_image(self,image, threshold=0.5):
        image[image < threshold] = 0
        image[image >= threshold] = 1
        return image

    def read_label(self,label_path,resize = None,threshold = True):
        
        full_path = self.__fullpath(label_path)
        image = io.imread(full_path,as_grey=True)
        
        if resize != None:
            image = transform.resize(image,resize)

        if(threshold):
            image = self.threshold_image(image)

        assert np.logical_or((image == 1), (image == 0)).all(), "expected 0 or 1 in binary label"

        return image
    
    def construct_image_path(self,image_path,offset = None):
        name,frame_no = self.split_path(image_path)
        
        if offset !=None:
            frame_no = (frame_no + offset) if (frame_no + offset) >=0 else 0
        
        return self.image_path(name, frame_no) 
    
    def construct_label_path(self,image_path,offset = None):
        name,frame_no = self.split_path(image_path)
        
        if offset !=None:
            frame_no = (frame_no + offset) if (frame_no + offset) >=0 else 0
        
        return self.label_path(name, frame_no)    
     
    def diff_labels(self,label_file1,label_file2,size):
        label1 = self.read_label(label_file1,size)
        label2 = self.read_label(label_file2,size)
    
        return label1 - label2
    
    def get_weight_map(self,image_path,resize = None):
        
        label_path1 = self.construct_label_path(image_path)
        label_path2 = self.construct_label_path(image_path,offset=-1)
    
        label1 = self.read_label(label_path1, resize)
        label2 = self.read_label(label_path2, resize)
    
        return self.__get_weight_map(label1, label2)
    
    def __get_weight_map(self,label1,label2):
        
        diffmap = label1 - label2
        diffmap = np.absolute(diffmap)
        diffmap[diffmap>0] = 1    
        
        # Expand diff to neighbouring regions
        diffmap = morphology.dilation(diffmap,np.ones([5,5]))
        
        zeros = np.where(diffmap==0)[0]
        ones = np.where(diffmap==1)[0]
        factor = 5#(zeros.shape[0]/ones.shape[0]) if ones.shape[0]!=0 else 1
        diffmap = factor*diffmap
        diffmap[diffmap==0]=1
        return diffmap
    
if __name__ == '__main__':
    helper = DataAccessHelper()  
    img_path = helper.label_path("bear", 76)
    full_path = os.path.join(helper.base_dir, img_path)
    image = Image.open(full_path)
    image = np.array(image)
    #diff = image[:,:,0] - image[:,:,1]
    image_io = skimage.img_as_ubyte(io.imread(full_path, as_grey=True))
    print(np.unique(image_io))
    print(np.unique(image[:,:,0]))
    print(np.unique(image[:,:,1]))


    print(np.unique(diff))
    #img = helper.read_image(img_path,[480,854])
    #print(img.shape)
    
