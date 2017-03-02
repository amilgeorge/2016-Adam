import os
from dataprovider.davis import DataAccessHelper
import numpy as np
import matplotlib.pyplot as plt
from common.logger import getLogger
from skimage import io, transform
import skimage
from mpl_toolkits.axes_grid1 import make_axes_locatable



IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480

davis = DataAccessHelper()
logger = getLogger()

def save_agg_vis_frame(out_seq_dir,seq_name,frame_no, src_dirs,src_frames,mean_frame,img):
    os.makedirs(out_seq_dir,exist_ok = True)
    out_fig_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    src_dir_names = [name.split(os.sep)[-2] for name in src_dirs]
    fig = plt.figure(figsize=(16,10))

    ax = plt.subplot(2,3,1)
    plt.imshow(img)

    ax = plt.subplot(2,3,2)
    im = ax.imshow(mean_frame)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = plt.subplot(2,3,4)
    im = ax.imshow(src_frames[0])
    ax.set_title(src_dir_names[0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = plt.subplot(2,3,5)
    im = ax.imshow(src_frames[1])
    ax.set_title(src_dir_names[1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = plt.subplot(2,3,6)
    im = ax.imshow(src_frames[2])
    ax.set_title(src_dir_names[2])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fig.tight_layout()
    fig.savefig(out_fig_path, dpi=fig.dpi)
    plt.close(fig)

def threshold_image(prediction,threshold=0.5):
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    return prediction

def save_image(out_dir,sequence_name,frame_no,img):
    out_seq_dir = os.path.join(out_dir,sequence_name)
    os.makedirs(out_seq_dir,exist_ok = True)
    out_path = os.path.join(out_seq_dir,'{0:05}.png'.format(frame_no))
    io.imsave(out_path, img)

def agg_seq(src_dirs,sequence_name,out_dir):
    # Mask output dir
    mask_out_dir = os.path.join(out_dir, '480p')
    agg_vis_dir = os.path.join(out_dir, 'agg-vis',sequence_name)

    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path, [IMAGE_HEIGHT, IMAGE_WIDTH]) * 255
    save_image(mask_out_dir, sequence_name, min(frames), davis.read_label(label_path))

    for frame_no in range(min(frames) + 1, max(frames) + 1):
        src_frames = []
        image_path = davis.image_path(sequence_name, frame_no)
        img = davis.read_image(image_path, [IMAGE_HEIGHT, IMAGE_WIDTH])

        for sd in src_dirs:
            pfile = os.path.join(sd,sequence_name,'{0:05}.npy'.format(frame_no))
            pmap = np.load(pfile)
            src_frames.append(pmap)

        mean_frame = np.mean(np.dstack(src_frames),axis=2)
        save_agg_vis_frame(agg_vis_dir, sequence_name, frame_no, src_dirs,
                           src_frames, mean_frame, img)

        pred_mask = threshold_image(mean_frame)
        img_shape = davis.image_shape(image_path)[0:2]
        pred_mask = transform.resize(pred_mask, img_shape)
        save_image(mask_out_dir, sequence_name, frame_no, skimage.img_as_ubyte(pred_mask))




def aggregate_sequences(src_dirs,out_dir):
    test_sequences = davis.test_sequence_list()+davis.train_sequence_list()

    for seq in test_sequences:
        logger.info('aggregating results for sequence: {}'.format(seq))
        agg_seq(src_dirs,seq,out_dir)

if __name__ == '__main__':

    out_dir = 'agg_half2-O1O5O10'
    out_dir = os.path.join('../Results',out_dir)
    src_dirs =['segnetvggwithskip-half2-wl-osvos-O1-1-1',
                  'segnetvggwithskip-half2-wl-osvos-O5-1-O5-1',
                  'segnetvggwithskip-half2-wl-osvos-O10-1-O10-1']

    src_dirs = [os.path.join('../Results',sd,'prob_maps') for sd in src_dirs]
    aggregate_sequences(src_dirs,out_dir)


