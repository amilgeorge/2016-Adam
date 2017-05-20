'''
Created on Dec 11, 2016

@author: george
'''
from dataprovider.davis import DataAccessHelper
import numpy as np
import matplotlib.pyplot as plt
from dataprovider.transformer import TransformParams,ImageRandomTransformer
from PIL import Image
import numpy.random as random
import dataprovider.inputhelper as inputhelper
from dataprovider.inputprovider import DataBatch,DB
import os
from common.logger import getLogger
from common.diskutils import ensure_dir
from dataprovider.preprocess import vgg_preprocess

logger = getLogger() 
 
class SeqDataIterator:
        
        def __init__(self,db, batch_size=1,is_training=True):
            self.db=db
            self.is_training = is_training
            numImages = self.db.images.shape[0]
            self.resize = list(self.db.images.shape[1:3])
            self.sequence_info = list(range(numImages))
            self.index = 0
            self.batch_size = batch_size
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.index < len(self.sequence_info):
                
                
                toIndex = self.index+self.batch_size
                
                selected_indexes = self.sequence_info[self.index:toIndex]
                
                # Read images and labels 
                images = np.zeros([self.batch_size]+self.resize+[7],dtype = np.float32)
                labels = np.zeros([self.batch_size]+self.resize)
                
                for i,idx in enumerate(selected_indexes):
                    img,label = self.get_input_ch7(idx)
                    images[i,:,:,:] = img[0,:,:,:]
                    labels[i,:,:] = label[0,:,:] 
                
                

                images = vgg_preprocess(images)
                                
                # Prepare data batch
                batch = DataBatch()
                batch.images = images
                batch.labels = labels
                self.index += self.batch_size
                return batch
            else:       
                raise StopIteration()  
        
        def get_prev_image_index(self,index):
            prev_index = (index -1) if (index -1) >=0 else 0
            return prev_index

        def get_input_ch7(self,index):
            prev_index = self.get_prev_image_index(index)
            image = self.db.images[index,:,:,:] * 255
            label = self.db.labels[index,:,:]
            prev_image = self.db.images[prev_index,:,:,:] * 255
            prev_mask = self.db.labels[prev_index,:,:]
            
            input_img = inputhelper.prepare_input_ch7(image, prev_mask, prev_image)
            return input_img,np.expand_dims(label, axis=0)
            
                      
class FineTuneDataProvider:
    """
    Data provider for fine tuning network per sequence
    """
    RESIZE_HEIGHT = 224
    RESIZE_WIDTH  = 224  
    def __init__(self,sequence, batch_size,is_training=True):
        self.sequence = sequence
        self.resize = [self.RESIZE_HEIGHT,self.RESIZE_WIDTH]

        self.is_training = is_training
        self.davis = DataAccessHelper()
        self.batch_size = batch_size
        self.db= self.load_db()
        self.val_db = self.load_val_db()
        self.fg_crop,self.fg_mask = self.get_fg_object()
        self.fg_transformer = ImageRandomTransformer(self._get_fg_transform_config())
        self.bg_transformer = ImageRandomTransformer(self._get_bg_transform_config())
        self.prev_image_transformer = ImageRandomTransformer(self._get_prev_image_transform_config())
        
        
    def __iter__(self):
        return self
    
    def _get_fg_transform_config(self):
        config ={}
        config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] =[-30,30]
        config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = 15
        return config
    
    def _get_bg_transform_config(self):
        config ={}
        config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] =[-10,10]
        config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = 5
        config[ImageRandomTransformer.CONFIG_SHEAR_RANGE] = [-5,5]
        return config
    
    def _get_prev_image_transform_config(self):
        config ={}
        config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] =[-20,20]
        config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = 5
        config[ImageRandomTransformer.CONFIG_FLIP_H] = False 
        return config
    
    def load_db(self):
        num_images = 1
        
        db = DB()
        db.images = np.zeros([num_images,self.RESIZE_HEIGHT,self.RESIZE_WIDTH,3])
        db.labels = np.zeros([num_images,self.RESIZE_HEIGHT,self.RESIZE_WIDTH])
        
        image_file = self.davis.image_path(self.sequence, 0)
        label_file = self.davis.label_path(self.sequence, 0)
        
        db.images[0,:,:,:] = self.davis.read_image(image_file, self.resize)
        db.labels[0,:,:] = self.davis.read_label(label_file, self.resize).astype(np.uint8)
        return db
    
    
    def load_val_db(self):
        all_frame_nums = self.davis.all_frames_nums(self.sequence)
        num_images = len(all_frame_nums)
        db = DB()
        db.images = np.zeros([num_images,self.RESIZE_HEIGHT,self.RESIZE_WIDTH,3])
        db.labels = np.zeros([num_images,self.RESIZE_HEIGHT,self.RESIZE_WIDTH])
        
        for i,num in enumerate(all_frame_nums):
            image_file = self.davis.image_path(self.sequence, num)
            label_file = self.davis.label_path(self.sequence, num)
            
            db.images[i,:,:,:] = self.davis.read_image(image_file, self.resize)
            db.labels[i,:,:] = self.davis.read_label(label_file, self.resize).astype(np.uint8)
        
        return db
    
    def get_bg(self):
        bg = self.db.images[0,:,:,:]
        bg[self.db.labels[0,:,:]>0,:] = 0
        return bg
    
    def get_fg_object(self):
        image = self.db.images[0,:,:,:]
        label = self.db.labels[0,:,:]
        min_r,min_c,max_r,max_c = self.bbox(label)
        fg_crop = np.copy(image[min_r:max_r,min_c:max_c])
        fg_mask = np.copy(label[min_r:max_r,min_c:max_c])
        return fg_crop,fg_mask
        
    def bbox(self,label):
        indices  = np.where(label>0)
        min_r,min_c = np.min(np.asarray(indices),axis=1)
        max_r,max_c = np.max(np.asarray(indices),axis=1)
        return min_r,min_c,max_r,max_c

    def paste(self,bg,fg,fg_mask,loc):
    
        bg =np.uint8(bg*255)
        fg = np.uint8(fg*255)
        bg_pil = Image.fromarray(bg,'RGB')
        #bg_pil.show()   
        fg_pil = Image.fromarray(fg,'RGB')
        mask_pil = Image.fromarray(fg_mask*255)
        
        bg_shape = bg.shape[0:2]
        fg_shape = fg.shape[0:2]
        
        label = np.zeros(bg_shape,dtype=np.uint8)
        label[loc[1]:loc[1]+fg_shape[0],loc[0]:loc[0]+fg_shape[1]] = fg_mask
        bg_pil.paste(fg_pil,loc,mask_pil)
        return np.array(bg_pil,np.uint8),label
    
    
    def get_image(self):
        curr_img, mask = self.get_curr_image()     
        prev_img, prev_mask = self.get_prev_image(curr_img, mask)
        
        prev_img = prev_img*255
        input_img = inputhelper.prepare_input_ch7(curr_img, prev_mask, prev_img)
        
        return input_img,np.expand_dims(mask, axis=0)
        
    
    def get_prev_image(self,curr_image,label):
        img_stack = np.concatenate((curr_image,np.expand_dims(label, axis=2)),axis=2)
        prev_img_stack,_ = self.prev_image_transformer.get_random_transformed(img_stack)
        
        return prev_img_stack[:,:,0:3],prev_img_stack[:,:,3]
        
    def get_curr_image(self):
        
        # create current image
        bg = self.get_bg()
        trans_bg,_ = self.bg_transformer.get_random_transformed(bg)
        
        fg = np.concatenate((self.fg_crop,np.expand_dims(self.fg_mask,axis=2)),axis = 2)
        trans_fg,_ = self.fg_transformer.get_random_transformed(fg)
        
        bg_height,bg_width=bg.shape[0:2]
        fg_height,fg_width=fg.shape[0:2]
        x = random.randint(0,bg_width-fg_width)
        y = random.randint(0,bg_height-fg_height)
        fg_mask = np.uint8(trans_fg[:,:,3])
        #fg_mask[fg_mask>0] =1
        img,label = self.paste(trans_bg, trans_fg[:,:,0:3],fg_mask, loc=(x,y))
        
        # create  from prev image from current image
        
        return img,label
    
    def get_next_minibatch(self):
        
        images = np.zeros([self.batch_size]+self.resize+[7],dtype = np.float32)
        labels = np.zeros([self.batch_size]+self.resize)
        
        for i in range(self.batch_size):
            img,label = self.get_image()
            images[i,:,:,:] = img[0,:,:,:]
            labels[i,:,:] = label[0,:,:]  
        
        images = vgg_preprocess(images)
        
        batch = DataBatch()
        batch.images = images
        batch.labels = labels
        
        return batch
            
    def val_seq_batch_itr(self):
        return SeqDataIterator(self.val_db)


        
def unit_test(out_dir='/usr/stud/george/test_fine_tune',seq='bear',num_imgs=100):
    data = FineTuneDataProvider(seq,16)
    out_seq_dir = os.path.join(out_dir,seq)
    ensure_dir(out_seq_dir)
    for i in range(num_imgs):
        a,mask = data.get_image()
        plt.subplot(2,2,1)
        frame = plt.gca()
        frame.axes.set_title('Img')
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.imshow(np.uint8(a[0,:,:,0:3]))
        plt.subplot(2,2,2)
        frame = plt.gca()
        frame.axes.set_title('Prev Img')
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.imshow(np.uint8(a[0,:,:,4:7]))
        plt.subplot(2,2,3)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.imshow(mask[0,:,:])
        plt.subplot(2,2,4)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.imshow(a[0,:,:,3])
        img_path = os.path.join(out_seq_dir,"{}.jpg".format(i))
        plt.savefig(img_path)
        logger.info('Generated to {}'.format(img_path))
        
        

if __name__ == '__main__':
 
    unit_test(seq = 'paragliding-launch')
#    dp = FineTuneDataProvider("bear",16)
#     iterator = dp.val_seq_batch_itr()  
#     for i,b in enumerate(iterator):
#         pass  
#     print("Done")
    
    

    
    
        