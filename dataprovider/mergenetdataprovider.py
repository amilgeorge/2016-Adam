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
from dataprovider.preprocess import vgg_preprocess, reverse_vgg_preprocess
from dataprovider.davis import DataAccessHelper
from dataprovider.davis_cached import DataAccessHelper as DataAccessHelperCached

from dataprovider.transformer_rand import ImageRandomTransformer
from numpy import dtype, size
import random


import matplotlib.pyplot as plt
from binstar_client.utils.notebook.data_uri import Image
from dataprovider import inputhelper
from common import diskutils
import glob


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 854



    
class InputProvider:
    
    BASE_DIR = os.path.join('/work/george','DAVIS')
    IMAGESETS = os.path.join('ImageSets','480p')

    NUM_CHANNELS = 7

    
    def __init__(self, branch1_offset,branch2_offset):
        
        self.davis = DataAccessHelper()
        self.resize = [IMAGE_HEIGHT,IMAGE_WIDTH]
        self.branch1_offset = branch1_offset
        self.branch2_offset = branch2_offset

        #self.val_set_info = self.val_set_info[:419,:]
        train_db_name = 'train'
        val_db_name = 'val'

        self.train_file_list = self.load_train_file_list()
        #print("loading db for branch1")
        #self.db_branch1 = self.loaddb(branch1_offset,train_db_name)

        #print("loading db for branch2")
        #self.db_branch2 = self.loaddb(branch2_offset,train_db_name)

    
    class DB: 
        def __init__(self,):
            pass
           
            
    class DataBatch:pass
    
    class DataIterator:
        
        def __init__(self,data_list,branch1_offset,branch2_offset, resize, batch_size,is_training=True):
            self.davis = DataAccessHelperCached([IMAGE_HEIGHT,IMAGE_WIDTH])
            self.data_list= data_list
            self.branch1_offset = branch1_offset
            self.branch2_offset = branch2_offset

            self.is_training = is_training
            numImages = len(data_list)
            self.img_size = resize
            #self.weights_size = list(self.db.weights.shape[1:3])
            self.img_channels = InputProvider.NUM_CHANNELS
            self.sequence_info = np.random.permutation(list(range(numImages)))
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

        def read_input_img(self,seq,frame_no,offset):
            assert offset<=0, "offset should be negative"

            rgb_path = self.davis.image_path(seq,frame_no)

            if offset == 0:
                prev_rgb_path = self.davis.image_path(seq, 0)
                prev_mask_path = self.davis.label_path(seq, 0)
            else:
                prev_rgb_path = self.davis.construct_image_path(rgb_path, offset=offset)
                prev_mask_path = self.davis.construct_label_path(rgb_path, offset=offset)

            rgb = self.davis.read_image(rgb_path, [IMAGE_HEIGHT, IMAGE_WIDTH])
            prev_mask = self.davis.read_label(prev_mask_path, [IMAGE_HEIGHT, IMAGE_WIDTH])
            prev_rgb = self.davis.read_image(prev_rgb_path, [IMAGE_HEIGHT, IMAGE_WIDTH])

            #print(np.unique(prev_mask))
            inp_img = inputhelper.prepare_input_img(rgb, prev_mask, prev_rgb)

            return inp_img[0, :, :, :]

        def __next__(self):
            if self.index < len(self.sequence_info):
                
                
                toIndex = self.index+self.batch_size
                
                selected_indexes = self.sequence_info[self.index:toIndex]
                
                # Read images and labels 
                images_branch1 = np.zeros([self.batch_size]+self.img_size+[self.img_channels],dtype = np.float32)
                images_branch2 = np.zeros([self.batch_size]+self.img_size+[self.img_channels],dtype = np.float32)
                labels = np.zeros([self.batch_size]+self.img_size)
                weights = np.empty([self.batch_size]+self.img_size,dtype=np.float32)

                for i,idx in enumerate(selected_indexes):
                    seq,frame_no = self.data_list[idx]

                    label_path = self.davis.label_path(seq,frame_no)
                    label = self.davis.read_label(label_path,self.img_size)

                    label_dim = np.expand_dims(label,axis=2)

                    image_branch1 = self.read_input_img(seq,frame_no,offset= -1*self.branch1_offset)
                    image_branch2 = self.read_input_img(seq,frame_no,offset= -1*self.branch2_offset)

                    stked = np.concatenate((image_branch1,image_branch2,label_dim),axis = 2)
                    #import pdb
                    #pdb.set_trace()
                    stked,_ = self.transformer.get_random_transformed(stked)
                    image_branch1 = stked[:,:,0:7]
                    image_branch2 = stked[:, :, 7:14]
                    if self.is_training:
                        image_branch1 = self.process_prev_mask_in_image(image_branch1)
                        image_branch2 = self.process_prev_mask_in_image(image_branch2)

                    
                    label = np.uint8(stked[:,:,14])
                    ## Label as changes
                    images_branch1[i,:,:,:]=image_branch1
                    images_branch2[i,:,:,:]=image_branch2
                    labels[i,:,:]=label

                    if ((np.count_nonzero(images_branch1[i,:, :, 3]) == 0) and (np.count_nonzero(images_branch2[i,:, :, 3]) == 0)):
                        print("warning :imgprovider previmg all zeros")
                        labels[i, :, :] = 0

                    #weights[i,:,:] = inputhelper.get_distance_transform(label)

                    weights[i,:,:] = inputhelper.get_weights_classwise_osvos(label)
                    #weights[i,:,:] = inputhelper.get_weights_osvos_distance(label)

                    #weights[i,:,:] = inputhelper.get_weights_classwise2(label,factor=3)






                images_branch1 = (images_branch1*255)
                images_branch1 = vgg_preprocess(images_branch1)
                images_branch2 = (images_branch2 * 255)
                images_branch2 = vgg_preprocess(images_branch2)

                # Prepare data batch
                batch = InputProvider.DataBatch()
                batch.images_branch1 = images_branch1
                batch.images_branch2 = images_branch2
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
            options = [DILATE,NONE_]
            #options = [NONE_]
            ch = random.choice(options)
            if ch == ERODE:
                sz = random.randint(2,4)
                new_mask = morphology.erosion(prev_mask,np.ones([sz,sz]))
                return new_mask
            elif ch == DILATE:

                sz = random.randint(2,10)
                new_mask = morphology.dilation(prev_mask,np.ones([sz,sz]))
                return new_mask
            else :
                return prev_mask
            
    def sequence_batch_itr(self, batch_size):
        return self.DataIterator(self.train_file_list,self.branch1_offset,self.branch2_offset,self.resize, batch_size)
    
    def val_seq_batch_itr(self, batch_size):
        return self.DataIterator(self.validation_db, batch_size,is_training=False)
    
    def getPrevMaskFile(self,labelFile):
        m=re.match(r"(/.*/.*/.*/)(.*).png",labelFile)
        prefix = m.group(1)
        frameNo = int(m.group(2))
        maskFrameNo = frameNo-1 if frameNo -1 >=0 else 0
        
        
        prevMaskFile='{0}{1:05}.png'.format(prefix,maskFrameNo)
        return prevMaskFile



    def loaddb(offset,name):
        if offset==0:
            print("Loading DB0 ")
            base_dir = os.path.join('/work/george/cache/db-480p/0/', name)
            images_cached_file = os.path.join(base_dir, 'images_db.npy')
            labels_cached_file = os.path.join(base_dir, 'labels_db.npy')
        else:
            print("Loading DB{} ".format(offset))
            base_dir = os.path.join('/work/george/cache/db-480p/offset-{}/'.format(offset), name)
            images_cached_file = os.path.join(base_dir, 'images_db.npy')
            labels_cached_file = os.path.join(base_dir, 'labels_db.npy')

        print("loading readonly memmap...")

        db = InputProvider.DB()

        db.images = np.load(images_cached_file, mmap_mode='r')
        db.labels = np.load(labels_cached_file, mmap_mode='r')

        print("db memmap loaded...")

        return db

    def createDB(self,data_set_info,name='',offsets=[]):
        print("Creating DB with offsets: {}".format(offsets))
        #offsets = list(range(1,7))
        #import cPickle as pickle
        #cached_db = '/work/george/cache/davisdb.pickle'
        offset_string = "-".join(str(x) for x in offsets)
        base_dir = os.path.join('/work/george/cache/db-480p/offset-{}/'.format(offset_string),name)
        images_cached_file = os.path.join(base_dir,'images_db.npy')
        labels_cached_file = os.path.join(base_dir,'labels_db.npy')
        num_images = data_set_info.shape[0]

        if not (os.path.isfile(images_cached_file) and os.path.isfile(labels_cached_file)):

            print("cached db not found. preparing memmap ...")

            diskutils.ensure_dir(base_dir)

            if(os.path.isfile(images_cached_file)):
                print("please clean cache file : ",images_cached_file)
                return
            if (os.path.isfile(labels_cached_file)):
                print("please clean cache file : ", labels_cached_file)
                return

            #Prepare temporary files for reading
            tmp_files_dir = os.path.join(base_dir,"temp")
            diskutils.ensure_dir(tmp_files_dir)

            self.prepare_temp_files(data_set_info,offsets,tmp_files_dir)

            # Create the database
            all_files = glob.glob(os.path.join(tmp_files_dir,"*.npz"))
            num_db_images = len(all_files)
            #images_mmap = np.memmap(images_cached_file, dtype='float32', mode='w+',
            #                        shape=tuple([num_db_images] + self.resize + [SampleInputProvider.NUM_CHANNELS]))
            #labels_mmap = np.memmap(labels_cached_file, dtype='float32', mode='w+',
            #                        shape=tuple([num_db_images] + self.resize))
            images_mmap = np.lib.format.open_memmap(images_cached_file, dtype='float32', mode='w+',
                                    shape=tuple([num_db_images] + self.resize + [InputProvider.NUM_CHANNELS]))
            labels_mmap = np.lib.format.open_memmap(labels_cached_file, dtype='float32', mode='w+',
                                    shape=tuple([num_db_images] + self.resize))

            self.create_memmap_db(tmp_files_dir,images_cached_file,labels_cached_file)





        print("loading readonly memmap...")

        db = InputProvider.DB()

        #db.images = np.memmap(images_cached_file, dtype='float32', mode='r', shape=tuple([num_db_images]+self.resize+[SampleInputProvider.NUM_CHANNELS]))
        #db.labels = np.memmap(labels_cached_file, dtype='float32', mode='r', shape=tuple([num_db_images] + self.resize))

        db.images = np.load(images_cached_file,mmap_mode='r')
        db.labels = np.load(labels_cached_file, mmap_mode='r')

        print("db memmap loaded...")

            
        return db

    def create_memmap_db(self,tmp_files_dir,images_cached_file,labels_cached_file):
        # Create the database
        print("creating memmap db...")

        all_files = glob.glob(os.path.join(tmp_files_dir, "*.npz"))
        num_db_images = len(all_files)
        # images_mmap = np.memmap(images_cached_file, dtype='float32', mode='w+',
        #                        shape=tuple([num_db_images] + self.resize + [SampleInputProvider.NUM_CHANNELS]))
        # labels_mmap = np.memmap(labels_cached_file, dtype='float32', mode='w+',
        #                        shape=tuple([num_db_images] + self.resize))
        images_mmap = np.lib.format.open_memmap(images_cached_file, dtype='float32', mode='w+',
                                                shape=tuple(
                                                    [num_db_images] + self.resize + [InputProvider.NUM_CHANNELS]))
        labels_mmap = np.lib.format.open_memmap(labels_cached_file, dtype='float32', mode='w+',
                                                shape=tuple([num_db_images] + self.resize))

        for i,f in enumerate(all_files):
            npzfile = np.load(f)
            image = npzfile['image']
            label = npzfile['label']

            images_mmap[i, :, :, :] = image
            labels_mmap[i, :, :] = label

        print("db:{} , num db images {}: ".format(tmp_files_dir, num_db_images))
        del images_mmap
        del labels_mmap
        print("saved db to disk...")

    def load_train_file_list(self):
        file_list = []
        for seq in self.davis.train_sequence_list():
            all_frames = self.davis.all_frames_nums(seq)
            for frame_no in all_frames:
                file_list.append((seq,frame_no))

        return file_list



    def read_image0(self, image_path, prev_mask=None):

        name, frame_no = self.davis.split_path(image_path)

        prev_rgb_path = self.davis.image_path(name, 0)


        if prev_mask is None:
            prev_mask_path = self.davis.label_path(name, 0)
            prev_mask = self.davis.read_label(prev_mask_path, self.resize)
            prev_mask = inputhelper.threshold_image(prev_mask)
            #print(np.unique(prev_mask))


        rgb = self.davis.read_image(image_path, self.resize)

        #prev_mask = np.expand_dims(prev_mask, axis=2)

        prev_rgb = self.davis.read_image(prev_rgb_path, self.resize)


        inp_img = inputhelper.prepare_input_img(rgb,prev_mask,prev_rgb)

        return inp_img[0,:,:,:]

    def read_image2(self,image_path, offset = 1,prev_mask = None):
    
        prev_mask_offset = -offset

        prev_rgb_path = self.davis.construct_image_path(image_path, offset= prev_mask_offset)
        
        rgb = self.davis.read_image(image_path, self.resize)
        
        if prev_mask is None:
            prev_mask_path = self.davis.construct_label_path(image_path, offset = prev_mask_offset)
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

def _debug(img_branch1,img_branch2,label,save_loc = None):
    plt.subplot(2,4,1)
    frame = plt.gca()
    frame.axes.set_title('Img branch1')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(np.uint8(img_branch1[:,:,0:3]))
    plt.subplot(2,4,2)
    frame = plt.gca()
    frame.axes.set_title('Prev Img branch1')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(np.uint8(img_branch1[:,:,4:7]))
    plt.subplot(2,4,3)
    frame = plt.gca()
    frame.axes.set_title('Img branch2')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(np.uint8(img_branch2[:,:,0:3]))
    plt.subplot(2,4,4)
    frame = plt.gca()
    frame.axes.set_title('Prev Img branch2')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(np.uint8(img_branch2[:,:,4:7]))
    plt.subplot(2,4,5)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(np.uint8(label))
    plt.subplot(2,4,6)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(img_branch1[:,:,3])
    plt.subplot(2,4,8)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.imshow(img_branch2[:,:,3])
    if save_loc:
        plt.savefig(save_loc)

    plt.close()
   
def test_gen_samples():

    provider = InputProvider(1 ,0)
    num=0
    dir = 'davis_merge_samp_0/'
    diskutils.ensure_dir(dir)
    max_num = 1000
    batch_size = 4
    while num < max_num:
        input_batch = provider.sequence_batch_itr(batch_size)

        for i, batch in enumerate(input_batch):

            if num > max_num:
                break

            imgs_branch1 = reverse_vgg_preprocess(batch.images_branch1)
            imgs_branch2 = reverse_vgg_preprocess(batch.images_branch2)

            for j in range(batch_size):
                if num > max_num:
                    break
                save_loc = os.path.join(dir, '{}.png'.format(num))

                _debug(imgs_branch1[j,:,:,:],imgs_branch2[j,:,:,:],batch.labels[j,:,:],save_loc)
                print (i, 'rgb files: ')
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
