'''
Created on Oct 7, 2016

@author: george
'''

import os
from skimage import io,transform,morphology
import numpy as np
import re
import yaml
from common import diskutils
import shutil,pickle
import matplotlib.pyplot as plt
from PIL import Image
from dataprovider import inputhelper
import skimage

class DB:
    pass

class DataAccessHelper(object):
    '''
    Helper class that provides convenient methods to access data from 
    davis data set
    '''


    def __init__(self,resize=[480,854],load_cached = True):
        '''
        Constructor
        '''
        self.image_set = '2016'
        self.base_dir = os.path.join('/work/george','DAVIS/')
        
        self.image_sets_dir = os.path.join(self.base_dir,'ImageSets')

        self.max_size = [480,854]
        self.image_path_prefix = os.path.join('JPEGImages','480p')
        self.annotations_prefix = os.path.join('Annotations','480p')
        self.db_info = os.path.join('Annotations','db_info.yml')
        self.cache_dir = os.path.join('/work/george','cache','davis{}_cache'.format(self.image_set))

        self.train_set_info = np.loadtxt(os.path.join(self.image_sets_dir,'480p','train.txt'), dtype=bytes,unpack=False).astype(str)
        self.val_set_info = np.loadtxt(os.path.join(self.image_sets_dir,'480p','val.txt'), dtype=bytes,unpack=False).astype(str)
        self.loaded = False
        if load_cached:
            self.db = self.load_cached_db()


    def read_db_info(self):
        with open(self.__fullpath(self.db_info),'r') as f:
            db_info = yaml.load(f)
            
            return db_info

    def cache_exists(self):
        index_dict_file = os.path.join(self.cache_dir, 'info_dict.p')

        return os.path.isfile(index_dict_file)

    def __name_idx_dict(self):
        name_idx_map = {}
        print(self.train_sequence_list())
        print(self.test_sequence_list())
        for seq in self.train_sequence_list()+self.test_sequence_list():
            name_idx_map[seq] = {}

        return name_idx_map
    def load_cached_db(self):
        images_cached_file = os.path.join(self.cache_dir, 'davis_images.npy')
        labels_cached_file = os.path.join(self.cache_dir, 'davis_labels.npy')
        index_dict_file = os.path.join(self.cache_dir, 'info_dict.p')
        if not self.cache_exists():
            print ("preparing db cache for davis")
            if os.path.isdir(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            diskutils.ensure_dir(self.cache_dir)
            self.prepare_cache(images_cached_file,labels_cached_file,index_dict_file)

        db = DB()
        db.name_idx_dict = pickle.load(open(index_dict_file, "rb"))
        db.images = np.load(images_cached_file, mmap_mode='r')
        db.labels = np.load(labels_cached_file, mmap_mode='r')
        self.loaded = True
        return db

    def prepare_cache_2017(self,images_cached_file,labels_cached_file,index_dict_file):
        num_train_seqs = self.train_set_info.shape[0]
        num_val_seqs = self.val_set_info.shape[0]
        
        
        
        num_train_images = 4219
        num_val_images = 2023
        
        
        num_images = num_train_images + num_val_images
        
        images_mmap = np.lib.format.open_memmap(images_cached_file, dtype='uint8', mode='w+',
                                                shape=tuple(
                                                    [num_images] + self.max_size + [3]))
        labels_mmap = np.lib.format.open_memmap(labels_cached_file, dtype='uint8', mode='w+',
                                                shape=tuple([num_images] + self.max_size))

        print(self.train_set_info.shape)

        data_set_info = np.hstack((self.train_set_info,self.val_set_info))
        print (data_set_info)

        name_idx_dict = self.__name_idx_dict()
        i = 0
        for seq in data_set_info:
            first_frame_path = self.image_path(seq, 0)
            img_shape = self.image_shape(first_frame_path)
            name_idx_dict[seq]['im_info'] = img_shape[:2]
            print(img_shape[:2])
            for frame_no in self.all_frames_nums(seq):
                image_file = self.image_path(seq,frame_no)
                label_file = self.label_path(seq,frame_no)

                image = self.read_image_disk(image_file)
                label = self.read_label_disk(label_file)

                name_idx_dict[seq][frame_no] = i
                images_mmap[i, :, :img_shape[1], :] = image
                labels_mmap[i, :, :img_shape[1]] = label
                print("loading image into memmap seq:{} frame no:{} at index:{}".format(seq,frame_no,i))
                i += 1

        del images_mmap
        del labels_mmap
        pickle.dump(name_idx_dict, open(index_dict_file, "wb"))
        print("cached db to disk...")

    def prepare_cache(self,images_cached_file,labels_cached_file,index_dict_file):
        num_train_images = self.train_set_info.shape[0]
        num_val_images = self.val_set_info.shape[0]
        num_images = num_train_images + num_val_images
        images_mmap = np.lib.format.open_memmap(images_cached_file, dtype='uint8', mode='w+',
                                                shape=tuple(
                                                    [num_images] + self.max_size + [3]))
        labels_mmap = np.lib.format.open_memmap(labels_cached_file, dtype='uint8', mode='w+',
                                                shape=tuple([num_images] + self.max_size))

        data_set_info = np.vstack((self.train_set_info,self.val_set_info))

        name_idx_dict = self.__name_idx_dict()
        for i in range(num_images):
            image_file = data_set_info[i, 0]
            image_file = image_file[1:]
            label_file = data_set_info[i, 1]
            label_file = label_file[1:]
            print(label_file)
            seq, frame_no = self.split_path(image_file)

            image = self.read_image_disk(image_file)
            label = self.read_label_disk(label_file)
            name_idx_dict[seq][frame_no] = i
            images_mmap[i, :, :, :] = image
            labels_mmap[i, :, :] = label

        del images_mmap
        del labels_mmap
        pickle.dump(name_idx_dict, open(index_dict_file, "wb"))
        print("cached db to disk...")

    def train_sequence_list(self):
        if self.image_set == '2016':
            db_info = self.read_db_info()
            db_sequences = db_info['sequences']
            test_seqs = filter(lambda s: 'training' == s['set'] ,db_sequences)

            test_seq_names = [seq['name'] for seq in test_seqs]
            return test_seq_names
        else:
            return self.train_set_info.tolist()
    
    def test_sequence_list(self):
        if self.image_set == '2016':
            db_info = self.read_db_info()
            db_sequences = db_info['sequences']
            test_seqs = filter(lambda s: 'test' == s['set'] ,db_sequences)

            test_seq_names = [seq['name'] for seq in test_seqs]
            return test_seq_names
        else:
            return self.val_set_info.tolist()

        
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
        

        m=re.match(r"(.*)/(.*/)(.*)/(.*).(jpg|png)",path)
        
        sequence_name = m.group(3)
        frame_no = int(m.group(4))
        return sequence_name,frame_no
    
    def __fullpath(self,image_path): 
        
        return os.path.join(self.base_dir, image_path)
    
    def image_shape(self,image_path):
        img = io.imread(self.__fullpath(image_path))  
        return img.shape

    def get_cached_idx(self,seq,frame_no):
        return self.db.name_idx_dict[seq][frame_no]

    def get_orig_size(self,seq):
        return self.db.name_idx_dict[seq]['im_info']

    def read_image(self, image_path):
        seq,frame_no = self.split_path(image_path)
        idx = self.get_cached_idx(seq,frame_no)

        return self.db.images[idx,:,:,:]

    def read_label(self,label_path):
        seq, frame_no = self.split_path(label_path)
        idx = self.get_cached_idx(seq, frame_no)

        label = (self.db.labels[idx, :, :] == 255)
        return label

        
    def read_image_disk(self, image_path):
        
        image = io.imread(self.__fullpath(image_path))       
        return image

    def threshold_image(self,image, threshold=0.5):
        image[image < threshold] = 0
        image[image >= threshold] = 1
        return image

    def read_label_disk(self,label_path):
        
        full_path = self.__fullpath(label_path)
        image =  skimage.img_as_ubyte(io.imread(full_path, as_grey=True))

        return image
    
    def num_objects(self,sequence_name):
        label_path = self.label_path(sequence_name, 0)

        if self.loaded:
            annotation = self.read_label_disk(label_path)
        else:
            annotation = self.read_label(label_path)
            
        return np.max(annotation)
        
                
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

    def get_input_ch7_gt_mask(self,seq,frame_no,prev_frame_no_calculator):
        prev_frame_no = prev_frame_no_calculator(frame_no)
        prev_mask_path = self.label_path(seq, prev_frame_no)
        prev_mask = np.uint8(self.read_label(prev_mask_path)*255)
        return self.get_input_ch7_with_mask(seq,frame_no,prev_mask,prev_frame_no_calculator)

    def get_input_ch7_with_mask(self,seq,frame_no,prev_mask,prev_frame_no_calculator):

        prev_frame_no = prev_frame_no_calculator(frame_no)

        rgb_path = self.image_path(seq,frame_no)
        prev_rgb_path = self.image_path(seq, prev_frame_no)

        rgb = self.read_image(rgb_path)
        prev_rgb = self.read_image(prev_rgb_path)

        inp_img = inputhelper.prepare_input_img_uint8(rgb, prev_mask, prev_rgb)
        

        return inp_img[0, :, :, :]
    
if __name__ == '__main__':
    helper = DataAccessHelper([480,854])

    row_s= []
    col_s = []
    print(len(helper.train_sequence_list()))
    print(len(helper.test_sequence_list()))

    train_frames = 0
    for seq in helper.train_sequence_list():
        frame_nums = helper.all_frames_nums(seq)
        print(len(frame_nums),max(frame_nums))
        train_frames = train_frames + len(frame_nums)


    test_frames = 0
    for seq in helper.test_sequence_list():
        frame_nums = helper.all_frames_nums(seq)
        print(len(frame_nums),max(frame_nums))
        test_frames = test_frames + len(frame_nums)

    print('Train frames',train_frames)
    print('Val frames',test_frames)

    print(np.unique(np.array(row_s)))
    print(np.unique(np.array(col_s)))
    #img = helper.read_label_disk(img_path)
    #print(img.shape)
    #plt.figure()
    #plt.imshow(img)
    
