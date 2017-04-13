'''
Created on Dec 13, 2016

@author: george
'''

import numpy as np
from dataprovider.preprocess import vgg_preprocess
from skimage import io,transform,morphology
import scipy.ndimage as ndi
import os

def prev_mask_path(out_dir,seq_name, frame_no, offset):
    prev_frame_no = max(0,frame_no - offset)
    return mask_path(out_dir,seq_name,prev_frame_no)

def mask_path(out_dir,seq_name, frame_no):
    label_path = os.path.join(out_dir,seq_name, '{0:05}.png'.format(frame_no))
    return label_path


def read_label(label_path, resize=None,threshold = True):
    """

    :param label_path:
    :param resize:
    :param threshold:
    :return: Mask with 0 or 1 label
    """
    image = io.imread(label_path, as_grey=True)

    if resize != None:
        image = transform.resize(image, resize)

    assert (image <=1.0).all(), "expected values <=1"

    image = threshold_image(image)

    assert np.logical_or((image == 1), (image == 0)).all(), "expected 0 or 1 in binary mask"
    #image = image * 255

    return image

def threshold_image(prediction,threshold=0.5):
    prediction[prediction < threshold] = 0
    prediction[prediction >= threshold] = 1
    return prediction

def get_distance_transform(label):
    label_inv = 1 - label
    dist  = ndi.morphology.distance_transform_cdt(label_inv)
    return dist


def get_dist_mul_map(label,steps = 7):
    label_inv = 1 - label
    dist  = ndi.morphology.distance_transform_cdt(label_inv)
    max_dist = max(label.shape)

    step_dist = (max_dist + steps -1)/steps

    dist_map = np.zeros(label.shape)

    ones = np.where(label==1)
    dist_map[ones] = 1

    for i in range(1,steps+1):
        min_dist = (i-1) * step_dist
        max_dist = i * step_dist

        locs = np.where((dist>min_dist) & (dist <= max_dist))
        dist_map[locs] = i

    return dist_map

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


def get_dilated(label, sz=10):
    dil_label = morphology.dilation(label, np.ones([sz, sz]))
    return dil_label

def get_weights_osvos_distance(label,resize=None):
    osvos_weights = get_weights_classwise_osvos(label,resize)
    dist_mul = get_dist_mul_map(label)

    return np.multiply(osvos_weights,dist_mul)

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

    
def prepare_input_img(img,prev_mask,prev_img):
    """
    Prepare the input img for neural network. 
    All inputs must be of the same size
    Keyword arguments:
    img -- the current image frame (0-1 range RGB)
    prev_mask -- the previous evaluated mask (0-1)
    prev_img -- previous RGB (0-1 range)
    """
    assert np.logical_or((prev_mask == 1), (prev_mask == 0)).all(), "expected 0 or 1 in binary prev mask"
    assert (img >= 0).all() and (img<=1).all(), "expected img values range 0-255"
    assert (prev_img >= 0).all() and (prev_img<=1).all(), "expected prev_img values range 0-255"

    prev_mask = np.expand_dims(prev_mask,axis=2)

    #print(img.shape)
    #print(prev_mask.shape)
    #print(prev_rgb.shape)

    # Concatenate images
    inp_img =  np.concatenate((img, prev_mask,prev_img), 2)
    
    # Apend axis for batch size
    inp_img = np.expand_dims(inp_img,axis=0)    
    
    
    return inp_img



if __name__ == '__main__':
    l = np.zeros((2,3))
    l[0,1]=1
    w = get_weights_classwise(l)
