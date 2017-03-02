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
from common import diskutils
import glob


IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480

davis = DataAccessHelper()
def prepare_input_ch7(image_path,prev_mask):
    
    # Read image
    img = davis.read_image(image_path, [IMAGE_HEIGHT,IMAGE_WIDTH])
    img = img*255
    
    mask = np.expand_dims(prev_mask,axis=2)    
    
    # Read previous image
    prev_img_path = davis.construct_image_path(image_path, offset= -1)
    prev_img = davis.read_image(prev_img_path, [IMAGE_HEIGHT,IMAGE_WIDTH])
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
    
    def __init__(self,resize = [RESIZE_HEIGHT,RESIZE_WIDTH],is_dummy = False, is_coarse=False):
        
        self.davis = DataAccessHelper()
        self.resize = resize
        if is_coarse:
            self.weights_resize = [SampleInputProvider.COARSE_OUT_HEIGHT,SampleInputProvider.COARSE_OUT_WIDTH]
        else:
            self.weights_resize = resize
        
        self.train_set_info = np.loadtxt(os.path.join(self.BASE_DIR,\
                        self.IMAGESETS,'train.txt'), dtype=bytes,unpack=False).astype(str)
        self.val_set_info = np.loadtxt(os.path.join(self.BASE_DIR,\
                        self.IMAGESETS,'val.txt'), dtype=bytes,unpack=False).astype(str)
        #self.val_set_info = self.val_set_info[:419,:]
        train_db_name = 'train'
        val_db_name = 'val'
        if is_dummy:
            train_db_name = 'dummy_train'
            val_db_name = 'dummy_val'
            self.train_set_info = self.train_set_info[247:255,:]
            self.val_set_info = self.val_set_info[247:255,:]

        offsets = [5]

        #offsets = list(range(1,7))#[10]
        self.db = self.createDB(self.train_set_info,name = train_db_name,offsets=offsets)
        #self.db = self.create_train_DB2(train_db_name,offsets)

        #half_db_name = "segnetvggwithskip-half-wl-osvos-O10-1-O10-1"
        #prev_mask_dir = os.path.join("..","Results",half_db_name,"480p")
        #self.db = self.create_train_DB3(name=half_db_name,
        #            prev_mask_dir=prev_mask_dir,offsets=offsets)


        self.validation_db = self.createDB(self.val_set_info,name = val_db_name,offsets=offsets)
    
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
            #self.weights_size = list(self.db.weights.shape[1:3])
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
            config[ImageRandomTransformer.CONFIG_SCALE_FACTOR_RANGE] = [0.5,1.0]
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
                weights = np.empty([self.batch_size]+self.img_size,dtype=np.float32)
                for i,idx in enumerate(selected_indexes):
                    label_dim = np.expand_dims(self.db.labels[idx,:,:], axis=2)
                    stked = np.concatenate((self.db.images[idx,:,:,:],label_dim),axis = 2)
                    #import pdb
                    #pdb.set_trace()
                    stked,_ = self.transformer.get_random_transformed(stked)
                    image = stked[:,:,0:7]
                    if self.is_training:
                        image = self.process_prev_mask_in_image(image)
                    
                    label = np.uint8(stked[:,:,7]) 
                    ## Label as changes
                    prev_mask = np.uint8(stked[:,:,3])  
                    #label_changes = inputhelper.get_dilated(label)
                    #label = label_changes
#                     plt.figure()
#                     plt.subplot(3,1,1)
#                     plt.imshow(label)
#                     plt.subplot(3,1,2)
#                     plt.imshow(prev_mask)
#                     plt.subplot(3,1,3)
#                     plt.imshow(label_changes)
#                     plt.colorbar()
                    #self._debug(image,label)
                    images[i,:,:,:]=image
                    labels[i,:,:]=label
                    #weights[i,:,:] = inputhelper.get_distance_transform(label)

                    #weights[i,:,:] = inputhelper.get_weights_classwise_osvos(label)
                    weights[i,:,:] = inputhelper.get_weights_osvos_distance(label)

                    #weights[i,:,:] = inputhelper.get_weights_classwise2(label,factor=3)

                    #plt.figure()
                    #plt.subplot(2,1,1)
                    #plt.imshow(weights[i,:,:])
                    #plt.colorbar()
                    #plt.subplot(2,1,2)
                    #plt.imshow(label)
                    #plt.colorbar()
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

    def create_train_DB2(self, name='',offsets=[]):

        print("Creating DB2 with offsets:{} ".format(offsets))
        #offsets = [1] # list(range(1,7))
        # import cPickle as pickle
        # cached_db = '/work/george/cache/davisdb.pickle'
        offset_string = "-".join(str(x) for x in offsets)
        base_dir = os.path.join('/work/george/cache/db-half/offset-{}/'.format(offset_string), name)
        images_cached_file = os.path.join(base_dir, 'images_db.npy')
        labels_cached_file = os.path.join(base_dir, 'labels_db.npy')

        if not (os.path.isfile(images_cached_file) and os.path.isfile(labels_cached_file)):

            print("cached db not found. preparing memmap ...")

            diskutils.ensure_dir(base_dir)

            if (os.path.isfile(images_cached_file)):
                print("please clean cache file : ", images_cached_file)
                return
            if (os.path.isfile(labels_cached_file)):
                print("please clean cache file : ", labels_cached_file)
                return


            seqs = davis.train_sequence_list()
            tmp_files_dir = os.path.join(base_dir, 'temp')
            diskutils.ensure_dir(tmp_files_dir)

            for seq in seqs:
                frames = davis.all_frames_nums(seq)
                mid = int(max(frames)/2)
                print("train: for seq:{} using frames {} to {}".format(seq,min(frames)+1,mid))
                for frame_no in range(min(frames)+1,mid+1):
                    for offset in offsets:
                        imageFile = davis.image_path(seq, frame_no)
                        labelFile = davis.label_path(seq, frame_no)
                        # prevMaskFile = self.getPrevMaskFile(labelFile)
                        image = self.read_image2(imageFile, offset=offset)
                        if (np.count_nonzero(image[:, :, 3]) == 0):
                            print("skipping image all zeros: {}".format(imageFile))
                            continue
                        label = self.davis.read_label(labelFile, self.resize)
                        outfile = os.path.join(tmp_files_dir, 'file-{}-{}-o{}.npz'.format(seq, frame_no, offset))
                        np.savez(outfile, image=image, label=label)


            self.create_memmap_db(tmp_files_dir,images_cached_file,labels_cached_file)


        print("loading readonly memmap...")

        db = SampleInputProvider.DB()

        # db.images = np.memmap(images_cached_file, dtype='float32', mode='r', shape=tuple([num_db_images]+self.resize+[SampleInputProvider.NUM_CHANNELS]))
        # db.labels = np.memmap(labels_cached_file, dtype='float32', mode='r', shape=tuple([num_db_images] + self.resize))

        db.images = np.load(images_cached_file, mmap_mode='r')
        db.labels = np.load(labels_cached_file, mmap_mode='r')

        print("db memmap loaded...")

        return db

    def create_train_DB3(self, name='',prev_mask_dir = None, offsets = []):

        print("Creating DB3 with offsets {} prev_mask_folder:{}".format(offsets,prev_mask_dir))

        #offsets = [1] # list(range(1,7))
        offset_string = "-".join(str(x) for x in offsets)
        base_dir = os.path.join('/work/george/cache/db3/offset-{}/'.format(offset_string), name)
        images_cached_file = os.path.join(base_dir, 'images_db.npy')
        labels_cached_file = os.path.join(base_dir, 'labels_db.npy')

        if not (os.path.isfile(images_cached_file) and os.path.isfile(labels_cached_file)):

            print("cached db not found. preparing memmap ...")

            diskutils.ensure_dir(base_dir)

            if (os.path.isfile(images_cached_file)):
                print("please clean cache file : ", images_cached_file)
                return
            if (os.path.isfile(labels_cached_file)):
                print("please clean cache file : ", labels_cached_file)
                return


            seqs = davis.train_sequence_list()
            tmp_files_dir = os.path.join(base_dir,'temp')
            diskutils.ensure_dir(tmp_files_dir)

            for offset in offsets:
                for seq in seqs:
                    frames = davis.all_frames_nums(seq)
                    mid = int(max(frames)/2)
                    print("train: for seq:{} using frames {} to {}".format(seq,min(frames)+1,mid))
                    for frame_no in range(mid+1,max(frames)+1):

                            imageFile = davis.image_path(seq, frame_no)
                            labelFile = davis.label_path(seq, frame_no)
                            prev_mask_file = inputhelper.prev_mask_path(prev_mask_dir,seq,frame_no,offset)
                            prev_mask = inputhelper.read_label(prev_mask_file,self.resize)
                            image = self.read_image2(imageFile, offset=offset,prev_mask=prev_mask)
                            if (np.count_nonzero(image[:, :, 3]) == 0):
                                print("skipping image all zeros: {}".format(imageFile))
                                continue
                            label = self.davis.read_label(labelFile, self.resize)
                            outfile = os.path.join(tmp_files_dir, 'file-{}-{}-o{}.npz'.format(seq,frame_no, offset))
                            np.savez(outfile, image=image, label=label)

            self.create_memmap_db(tmp_files_dir,images_cached_file,labels_cached_file)

        print("loading readonly memmap...")

        db = SampleInputProvider.DB()

        # db.images = np.memmap(images_cached_file, dtype='float32', mode='r', shape=tuple([num_db_images]+self.resize+[SampleInputProvider.NUM_CHANNELS]))
        # db.labels = np.memmap(labels_cached_file, dtype='float32', mode='r', shape=tuple([num_db_images] + self.resize))

        db.images = np.load(images_cached_file, mmap_mode='r')
        db.labels = np.load(labels_cached_file, mmap_mode='r')

        print("db memmap loaded...")

        return db

    def prepare_temp_files(self,data_set_info,offsets,out_dir):
        num_images = data_set_info.shape[0]
        for i in range(num_images):
            for offset in offsets:
                imageFile = data_set_info[i, 0]
                imageFile = imageFile[1:]
                labelFile = data_set_info[i, 1]
                labelFile = labelFile[1:]
                # prevMaskFile = self.getPrevMaskFile(labelFile)
                image = self.read_image2(imageFile, offset=offset)
                if (np.count_nonzero(image[:, :, 3]) == 0):
                    print("skipping image all zeros: {}".format(imageFile))
                    continue
                label = self.davis.read_label(labelFile, self.resize)
                outfile = os.path.join(out_dir,'file-{}-o{}.npz'.format(i,offset))
                np.savez(outfile,image=image,label=label)


    def createDB(self,data_set_info,name='',offsets=[]):
        print("Creating DB with offsets: {}".format(offsets))
        #offsets = list(range(1,7))
        #import cPickle as pickle
        #cached_db = '/work/george/cache/davisdb.pickle'
        offset_string = "-".join(str(x) for x in offsets)
        base_dir = os.path.join('/work/george/cache/db/offset-{}/'.format(offset_string),name)
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
                                    shape=tuple([num_db_images] + self.resize + [SampleInputProvider.NUM_CHANNELS]))
            labels_mmap = np.lib.format.open_memmap(labels_cached_file, dtype='float32', mode='w+',
                                    shape=tuple([num_db_images] + self.resize))

            self.create_memmap_db(tmp_files_dir,images_cached_file,labels_cached_file)





        print("loading readonly memmap...")

        db = SampleInputProvider.DB()

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
                                                    [num_db_images] + self.resize + [SampleInputProvider.NUM_CHANNELS]))
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
    
   
    
if __name__ == '__main__':
    provider = SampleInputProvider(resize=[IMAGE_HEIGHT,IMAGE_WIDTH],is_coarse=False,is_dummy=True)
    for j in range(1,1000):
        input_batch = provider.sequence_batch_itr(16)

        for i, batch in enumerate(input_batch):
            print (i, 'rgb files: ')
    
    print("Done")
