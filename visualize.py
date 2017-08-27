'''
Created on Oct 18, 2016

@author: george
'''
from dataprovider.davis import DataAccessHelper
import os
from skimage import io, color
from PIL import  Image
import numpy as np

davis = DataAccessHelper()

def save_image(out_dir,sequence_name,frame_no,img):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    out_path = os.path.join(out_seq_dir,'{0:05}.jpg'.format(frame_no))
    io.imsave(out_path, img)

def overlay_mask(image,mask):
    
    ov_img = color.label2rgb(mask,image,colors=['red'],bg_label=0)
    return ov_img

def overlay_mask_color(image,mask,color=(255, 0, 0)):
    label = np.uint8(mask) * 123
    pil_img = Image.fromarray(image, mode='RGB')
    color_img = Image.new('RGB', pil_img.size,color)
    label = Image.fromarray(label, mode='L')
    # mask.show()

    pil_img.paste(color_img, (0, 0), label)

    return np.array(pil_img)

def threshold_image(prediction,threshold=0.5):
    prediction[prediction < threshold] = 0
    prediction[prediction >= threshold] = 1
    return prediction

def draw_masks(masks_dir,out_dir):
    sequences = davis.train_sequence_list()+davis.test_sequence_list()
    
    for seq in sequences:
        all_frames=davis.all_frames_nums(seq)
        print(seq,end="")
        for frame_no in all_frames:
            print(".",end="")
            image_path = davis.image_path(seq, frame_no)
            image = davis.read_image(image_path)
            if frame_no == 0:
                label_path = davis.label_path(seq, frame_no)
                mask = davis.read_label(label_path)
            else:
                abs_label_path = os.path.join(masks_dir, seq, '{0:05}.png'.format(frame_no))
                mask = io.imread(abs_label_path, as_grey=True)


            mask = threshold_image(mask,0.5)
            if frame_no ==0:
                ov_img = overlay_mask_color(image, mask,color=(0,255,0))
            else:
                ov_img = overlay_mask_color(image, mask)
            save_image(out_dir, seq, frame_no, ov_img)
        print("|")
            
if __name__ == '__main__':
    #dirs = ['agg_half2-O1O5O10']

    #dirs = ['s480pvgg-segnet_brn-daviscombo-O1-Plabel_to_dist-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iter-685000']
    dirs = ['s480pvgg-segnet_brn-daviscombo-O1-PNone-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iter-700000']

    for d in dirs:
        masks_dir = '../adam/test_out/{}/480p'.format(d)
        print ('processing for {}'.format(masks_dir))
        out_dir = masks_dir+'-vis'
        draw_masks(masks_dir,out_dir)
