import os
from dataprovider.davis_cached_2016 import DataAccessHelper
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

def db_eval_iou(annotation,segmentation):

	""" Compute region similarity as the Jaccard Index.

	Arguments:
		annotation   (ndarray): binary annotation   map.
		segmentation (ndarray): binary segmentation map.

	Return:
		jaccard (float): region similarity

 """

	annotation   = annotation.astype(np.bool)
	segmentation = segmentation.astype(np.bool)

	if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
		return 1
	else:
		return np.sum((annotation & segmentation)) / \
				np.sum((annotation | segmentation),dtype=np.float32)

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


def mask_path(out_dir,seq_name, frame_no):
    label_path = os.path.join(out_dir,seq_name, '{0:05}.png'.format(frame_no))
    return label_path

def read_label(label_path):
    image = io.imread(label_path, as_grey=True)
    return image

def upper_bound_seq(src_dirs,sequence_name,out_dir):
    # Mask output dir
    mask_out_dir = os.path.join(out_dir, '480p')
    agg_vis_dir = os.path.join(out_dir, 'agg-vis',sequence_name)


    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    prev_mask = davis.read_label(label_path) * 255
    save_image(mask_out_dir, sequence_name, min(frames), davis.read_label(label_path))


    for frame_no in range(min(frames) + 1, max(frames) + 1):
        print('computing seq:{} frame:{}'.format(sequence_name,frame_no))
        out_binary_list = []
        J_list = []
        image_path = davis.image_path(sequence_name, frame_no)
        label_path = davis.label_path(sequence_name, frame_no)

        img = davis.read_image(image_path)
        gt_label = davis.read_label(label_path)

        gt_binary = (gt_label > 0)

        for sd in src_dirs:
            out_label_path = mask_path(sd,sequence_name,frame_no)
            out_label = read_label(out_label_path)
            out_binary = out_label > 0
            j = db_eval_iou(gt_binary,out_binary)
            J_list.append(j)
            out_binary_list.append(out_binary)

        max_idx = np.argmax(J_list)
        print(J_list)
        print("Max index :{}",max_idx)
        mask_binary = out_binary_list[max_idx]
        pred  = skimage.img_as_ubyte(mask_binary)*255
        #print(np.unique(pred))
        save_image(mask_out_dir, sequence_name, frame_no, pred)




def process_sequences(src_dirs,out_dir):
    test_sequences = davis.test_sequence_list()+davis.train_sequence_list()

    for seq in test_sequences:
        logger.info('aggregating results for sequence: {}'.format(seq))
        upper_bound_seq(src_dirs,seq,out_dir)

if __name__ == '__main__':

    out_dir = 'upperbound_framewise_osvos_and_tpsm-dist'
    out_dir = os.path.join('../Results',out_dir)
    #src_dirs =['segnet480pvgg-wl-dp2-osvos-val-O0-4/iter-45000',
    #              'segnet480pvgg-wl-dp2-osvos-val-O1-3/iter-45000',
    #              ]

    #src_dirs = [os.path.join('test_out',sd,'480p') for sd in src_dirs]
    src_dirs = []
    #src_dirs.append("/usr/stud/george/workspace/adam/test_out/s480pvgg-davis2016-O1-osvosold-reg1e-4-mo<1e-2>-de-1/iter-500000/480p")
    #src_dirs.append("/work/george/DAVIS/Results/Segmentations/480p/OSVOS")
    src_dirs.append("/usr/stud/george/workspace/adam/test_out/s480pvgg-segnet_brn-daviscombo-O1-Plabel_to_dist-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/iter-685000/480p")

    src_dirs.append("/usr/stud/george/workspace/Results/OSVOS")

    process_sequences(src_dirs,out_dir)




