# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:04:31 2016

@author: george
"""

def _mean_image_subtraction(images, means):
  """Subtracts the given means from each image channel.

  Args:
    images: a array of size [batch_size, height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  
  if len(images.shape) != 4:
      raise ValueError('Input must be of size [batch_size, height, width, C>0]')
      
  num_channels = images.shape[-1]
      
  if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    
  preprocessed = images - means
    
  return preprocessed

def vgg_preprocess(images):
    """Preprocessing input for VGG net
    
    Args:
        images: Array of size [batch,height, width, C]. Images [0-255] scale
    
    Returns:
        the centered images
    
    """
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    _M_MEAN = 127
    
    channels = images.shape[3]
    if channels ==7:
        center = [_R_MEAN,_G_MEAN,_B_MEAN,_M_MEAN,_R_MEAN,_G_MEAN,_B_MEAN]
    else: 
        center = [_R_MEAN,_G_MEAN,_B_MEAN,_M_MEAN]
        
    return _mean_image_subtraction(images,center)