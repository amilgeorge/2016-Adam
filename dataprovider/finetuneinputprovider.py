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
class DB: pass

class DataBatch:pass
    
class FineTuneDataIterator:
    RESIZE_HEIGHT = 224
    RESIZE_WIDTH  = 224  
    def __init__(self,sequence, batch_size,is_training=True):
        self.sequence = sequence
        self.resize = [self.RESIZE_HEIGHT,self.RESIZE_WIDTH]

        self.is_training = is_training
        self.davis = DataAccessHelper()
        self.batch_size = batch_size
        self.db= self.load_db()
        self.fg_crop,self.fg_mask = self.get_fg_object()
        self.fg_transformer = ImageRandomTransformer(self._get_fg_transform_config())
        self.bg_transformer = ImageRandomTransformer(self._get_bg_transform_config())
        self.prev_image_transformer = ImageRandomTransformer(self._get_prev_image_transform_config())
        
        
    def __iter__(self):
        return self
    
    def _get_fg_transform_config(self):
        config ={}
        config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] =[-60,60]
        config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = 15
        return config
    
    def _get_bg_transform_config(self):
        config ={}
        config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] =[-20,20]
        config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = 5
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
    
    def crop_object(self,rgb,label):
        indices  = np.where(label>0)
        min_r,min_c = np.min(indices,axis=0)
        max_r,max_c = np.min(indices,axis=0)

    def paste(self,bg,fg,fg_mask,loc):
    
        bg =np.uint8(bg*255)
        fg = np.uint8(fg*255)
        bg_pil = Image.fromarray(bg,'RGB')
        bg_pil.show()   
        fg_pil = Image.fromarray(fg,'RGB')
        mask_pil = Image.fromarray(fg_mask*255)
        
        bg_shape = bg.shape[0:2]
        fg_shape = fg.shape[0:2]
        
        label = np.zeros(bg_shape,dtype=np.uint8)
        label[loc[1]:loc[1]+fg_shape[0],loc[0]:loc[0]+fg_shape[1]] = fg_mask
        bg_pil.paste(fg_pil,loc,mask_pil)
        return np.array(bg_pil,np.uint8),label
        
    def get_image(self):
        
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
        
    def __next__(self):
        pass


        

if __name__ == '__main__':
    data = FineTuneDataIterator("bear",14)
    
    image = data.db.images[0,:,:,:]
    label = data.db.labels[0,:,:]
    min_r,min_c,max_r,max_c = data.bbox(label)
    crop = image[min_r:max_r,min_c:max_c]
    bg = data.get_bg()
    fg = data.fg_crop 
    fg_mask = data.fg_mask
    #a,mask = data.paste(bg.astype(np.uint8), fg, fg_mask, (0,50))
    a,mask = data.get_image()

    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.show()
    print("Done")
    
    

    
    
        