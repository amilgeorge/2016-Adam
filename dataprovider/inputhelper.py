'''
Created on Dec 13, 2016

@author: george
'''

import numpy as np
from dataprovider.preprocess import vgg_preprocess
from skimage import io,transform,morphology



def get_weights_classwise(label,resize=None,factor = None):

    weight_map = np.zeros((label.shape))
    zeros = np.where(label==0)
    ones = np.where(label==1)
    if factor is None:
        factor = len(zeros[0])/len(ones[0]) if len(ones[0])!=0 else 1

    weight_map[zeros] = 1
    weight_map[ones] = factor
    return weight_map



def get_weights_classwise3(label,resize=None,factor = None):
    
    
    dil_label = morphology.dilation(label,np.ones([5,5]))
    weight_map = get_weights_classwise(dil_label, resize, factor)
    return weight_map


def get_weights_classwise_osvos(label, resize=None):
    label_resized = np.float64(label)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(label_resized)

    if resize is not None:
        label_resized = transform.resize(label_resized, resize, order=0)

    label_resized = np.round(label_resized)

    weight_map = np.zeros((label_resized.shape))
    zeros = np.where(label_resized == 0)
    ones = np.where(label_resized == 1)

    beta = len(zeros[0]) / (len(ones[0]) +len(zeros[0]))

    weight_map[zeros] = 1 - beta
    weight_map[ones] = beta
    # plt.figure()
    # plt.imshow(weight_map)
    # plt.colorbar()
    return weight_map

def get_weights_classwise2(label,resize=None,factor = None):

    label_resized = np.float64(label)
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(label_resized)
    
    if resize is not None:
        label_resized =  transform.resize(label_resized,resize,order=0)

    
    label_resized = np.round(label_resized)

    weight_map = np.zeros((label_resized.shape))
    zeros = np.where(label_resized == 0)
    ones = np.where(label_resized == 1)
    if factor is None:
        factor = len(zeros[0])/len(ones[0]) if len(ones[0])!=0 else 1

    weight_map[zeros] = 1
    weight_map[ones] = factor
    #plt.figure()
    #plt.imshow(weight_map)
    #plt.colorbar()
    return weight_map

def get_label_changes(label,prev_mask):
    changes = np.int8(label - prev_mask)
    changes = np.abs(changes)
    return np.uint8(changes)

    
def prepare_input_ch7(img,prev_mask,prev_img):
    """
    Prepare the input img for neural network. 
    All inputs must be of the same size
    Keyword arguments:
    img -- the current image frame (0-255 range RGB)
    prev_mask -- the previous evaluated mask (0-1)
    prev_img -- previous RGB (0-255 range)
    """
       
    prev_mask = np.expand_dims(prev_mask,axis=2)*255  
 
    # Concatenate images
    inp_img =  np.concatenate((img, prev_mask,prev_img), 2)
    
    # Apend axis for batch size
    inp_img = np.expand_dims(inp_img,axis=0)    
    
    
    return inp_img

if __name__ == '__main__':
    l = np.zeros((2,3))
    l[0,1]=1
    w = get_weights_classwise(l)