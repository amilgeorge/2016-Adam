'''
Created on Dec 12, 2016

@author: amilgeorge
'''
import random as random
import math
import numpy as np
from skimage import transform
from matplotlib import pyplot as plt

class TransformParams(object):
    
    def __init__(self):
        self.flip_v = False
        self.flip_h = False
        
        self.rotation = None
        self.scale_factor = None
        self.translation_factor = None
        self.shear = None
    
    def set_rotation_in_degrees(self,rot):
        if rot:
            self.rotation = np.deg2rad(rot)
    
    def set_scale_factor(self,s):
        assert len(s) == 2, "expected tuple or array of len 2"
        self.scale_factor = s
        
    def set_shear_in_degrees(self,angle):
        self.shear = np.deg2rad(angle)
    
    def set_flip_v(self,flip):
        self.flip_v = flip
    
    def set_flip_h(self,flip):
        self.flip_h = flip
    
    def set_translation_factor(self,t):
        assert len(t) == 2, "expected tuple or array of len 2"
        self.translation_factor = t    
    
    def get_translation(self,image_shape):
        assert len(image_shape)>=2, "ndims should be greater than or equal to 2"
        return (image_shape[0]*self.translation_factor[1],image_shape[1]*self.translation_factor[0])
        
class ImageRandomTransformer(object):
    '''
    Random Transformations for an image
    '''
    
    CONFIG_FLIP_V = 'flipV'
    CONFIG_FLIP_H = 'flipH'
    
    CONFIG_ROTATION_ANGLE_STEP = 'rotation_angle_interval'
    CONFIG_ROTATION_RANGE = 'rotation_range'
    
    CONFIG_SCALE_FACTOR_RANGE = 'scale_factor_range'
    CONFIG_SCALE_FACTOR_STEP = 'scale_factor_step'
    
    CONFIG_SHEAR_RANGE = 'shear_range'
    CONFIG_SHEAR_ANGLE_STEP = 'shear_step'
    

    CONFIG_TRANSLATION_FACTOR_RANGE = 'translation_range'
    CONFIG_TRANSLATION_FACTOR_STEP = 'translation_step'
    
    CONFIG_CONTRAST_ADJUSTMENT = 'contrast_adjustment'
    CONFIG_BRIGHTNESS_ADJUSTMENT = 'brightness_adjustment'
     
    
    
    

    def __init__(self,config_params={}):
        '''
        Constructor
        '''
    
        self.config = self.load_config(config_params)
        
        
    def load_config(self,config_params):
        config = self.default_config()
        config.update(config_params)
        return config
    
    def default_config(self):    
        default_config = {}
        # Flip Horizontal
        default_config[self.CONFIG_FLIP_H] = True
        # Flip Vertical
        default_config[self.CONFIG_FLIP_V] = False
        
        # Rotation Interval (in degrees)
        default_config[self.CONFIG_ROTATION_ANGLE_STEP] = 15  
        default_config[self.CONFIG_ROTATION_RANGE] = [0,180]
        
        # Scale Factors
        default_config[self.CONFIG_SCALE_FACTOR_RANGE] = [0.8,1.2]
        default_config[self.CONFIG_SCALE_FACTOR_STEP] = 0.1
        
        # Shear configuration (in degrees)
        default_config[self.CONFIG_SHEAR_ANGLE_STEP] = 1
        default_config[self.CONFIG_SHEAR_RANGE] = [-15,15]
        
        # Translation with respect to axis dimensions
        default_config[self.CONFIG_TRANSLATION_FACTOR_RANGE] = [-0.1,0.1]
        default_config[self.CONFIG_TRANSLATION_FACTOR_STEP] = 0.01
        
        return default_config


        
    def _generate_param_flip_v(self):
        if(self.config[self.CONFIG_FLIP_V]):
            return True if random.random() > 0.5 else False
        else:
            return None
        
    def _generate_param_flip_h(self):
        if(self.config[self.CONFIG_FLIP_H]):
            return True if random.random() > 0.5 else False
        else:
            return None
    
    def _generate_param_rotation(self):
        
        rot_range = self.config[self.CONFIG_ROTATION_RANGE]
        intvl_step = self.config[self.CONFIG_ROTATION_ANGLE_STEP]
        
        if not intvl_step:
            return None
        
        assert rot_range[0] <= rot_range[1],"rotation range invalid. please check"
        
        rot_deg = random.randrange(rot_range[0],rot_range[1]+intvl_step,intvl_step)
        return rot_deg  
    
    def _generate_param_scale(self):
        
        param_range = self.config[self.CONFIG_SCALE_FACTOR_RANGE]
        param_step = self.config[self.CONFIG_SCALE_FACTOR_STEP]
        
        if not param_step:
            return None
           
        assert param_range[0] <= param_range[1],"scale range invalid. please check"
        
        param_range_int_steps =  np.round((param_range[1] - param_range[0])/param_step)
        
        sx_step = random.randrange(0,param_range_int_steps + 1 ,1)
        sx = param_range[0]+sx_step*param_step
        
        scale_non_uniform = True if random.random() > 0.5 else False
        if scale_non_uniform:
            sy_step = random.randrange(0,param_range_int_steps + 1 ,1)
            sy = param_range[0]+sy_step*param_step
        else:
            sy = sx
            
        return sx,sy
    
    def _generate_param_shear(self):
        param_range = self.config[self.CONFIG_SHEAR_RANGE]
        param_step = self.config[self.CONFIG_SHEAR_ANGLE_STEP]
        
        if not param_step:
            return None
           
        assert param_range[0] <= param_range[1],"shear range invalid. please check"
        
        shear = random.randrange(param_range[0],param_range[1]+param_step,param_step)
            
        return shear
    
    def _generate_param_translate(self):
        
        param_range = self.config[self.CONFIG_TRANSLATION_FACTOR_RANGE]
        param_step = self.config[self.CONFIG_TRANSLATION_FACTOR_STEP]
        
        if not param_step:
            return None
           
        assert param_range[0] <= param_range[1],"translation range invalid. please check"
        
        param_range_int_steps =  np.round((param_range[1] - param_range[0])/param_step)
        tx_step = random.randrange(0,param_range_int_steps + 1 ,1)
        ty_step = random.randrange(0,param_range_int_steps + 1 ,1)

        tx = param_range[0]+tx_step*param_step

        ty = param_range[0]+ty_step*param_step
            
        return tx,ty    
        
    def _generate_random_transform_params(self):
        
        transform_params = TransformParams()
        transform_params.set_flip_h(self._generate_param_flip_h())
        transform_params.set_flip_v(self._generate_param_flip_v())
        transform_params.set_rotation_in_degrees(self._generate_param_rotation())
        transform_params.set_scale_factor(self._generate_param_scale())
        transform_params.set_translation_factor(self._generate_param_translate())
        transform_params.set_scale_factor(self._generate_param_scale())
        transform_params.set_shear_in_degrees(self._generate_param_shear())
        
        return transform_params
    
    def get_random_transformed(self,image):
            
        tp = self._generate_random_transform_params()
        im = self.get_transformed(image, tp)
        
        return im,tp
    
    def get_transformed(self,image,transform_params):
        
        t = transform_params.get_translation(image.shape)        
        tform = transform.AffineTransform(scale=transform_params.scale_factor,rotation=transform_params.rotation,
                                  shear=transform_params.shear,translation=t)
        
        warped = transform.warp(image,tform)
        if(transform_params.flip_h):
            warped = np.fliplr(warped)
        if(transform_params.flip_v):
            warped = np.flipud(warped)
       
        return warped
            
#if __name__ == '__main__':
#     t = ImageRandomTransformer()
#     DATA_PATH = "/media/linu/System2/py-r-fcn/py-R-FCN/data/products_online"
#     _classes = ('__background__', # always index 0
#                          'banana', 'chocolatebar')
#     p = t._generate_random_transform_params()
#     db = FgBgDS(DATA_PATH,_classes)
#     db.load()
#     dp = OnlineDataProvider("test",db)
#     img,bboxes,fg_idxs = dp.create_rand_image()
#     
#     plt.figure()
#     plt.subplot(1,10,1)
#     plt.imshow(img)
#     p_list=[]
#     for i in xrange(2,11):
#         plt.subplot(1,10,i)
#         warp_img,p=t.get_random_transformed(img)
#         p_list.append(p)
#         plt.imshow(warp_img)
#     plt.show()    
#     print t._generate_param_flip_v()