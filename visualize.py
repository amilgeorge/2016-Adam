'''
Created on Oct 18, 2016

@author: george
'''
from dataprovider.davis import DataAccessHelper
import os
from skimage import io, color

davis = DataAccessHelper()

def save_image(out_dir,sequence_name,frame_no,img):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    out_path = os.path.join(out_seq_dir,'{0:05}.jpg'.format(frame_no))
    io.imsave(out_path, img)

def overlay_mask(image,mask):
    
    ov_img = color.label2rgb(mask,image,colors=['red'],bg_label=0)
    return ov_img
    
def draw_masks(masks_dir,out_dir):
    sequences = davis.train_sequence_list()+davis.test_sequence_list()
    
    for seq in sequences:
        all_frames=davis.all_frames_nums(seq)
        
        for frame_no in all_frames:
            image_path = davis.image_path(seq, frame_no)
            image = davis.read_image(image_path)
            abs_label_path = os.path.join(masks_dir,seq,'{0:05}.png'.format(frame_no))
            mask = io.imread(abs_label_path,as_grey=True)
            ov_img = overlay_mask(image, mask)
            save_image(out_dir, seq, frame_no, ov_img)
            
            
if __name__ == '__main__':
    #dirs = ['agg_half2-O1O5O10']

    dirs = ['segnetvggwithskip-half2-wl-osvos-O1-1-1','segnetvggwithskip-half2-wl-osvos-O5-1-O5-1','segnetvggwithskip-half2-wl-osvos-O10-1-O10-1']

    for d in dirs:
        masks_dir = '../Results/{}/480p'.format(d)
        print ('processing for {}'.format(masks_dir))
        out_dir = masks_dir+'-vis'
        draw_masks(masks_dir,out_dir)
