'''
Created on Jan 19, 2017

@author: george
'''
from dataprovider.davis_cached_2016 import DataAccessHelper
from matplotlib import pyplot as plt
from skimage import morphology
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from dataprovider import inputhelper


davis = DataAccessHelper()

def func1():
    label_path = davis.label_path('bmx-bumps', 4)
    label = davis.read_label(label_path)


    e_mask = morphology.erosion(label, np.ones([5, 5]))
    d_mask = morphology.dilation(label, np.ones([5, 5]))

    fig, axes = plt.subplots(1,3)
    print(axes)
    axes[0].imshow(label,cmap='jet')
    axes[0].set_title('Original Mask')
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)

    axes[1].imshow(e_mask,cmap='jet')
    axes[1].set_title('Eroded Mask (s = 5)')
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)

    axes[2].imshow(d_mask,cmap='jet')
    axes[2].set_title('Dilated Mask (s = 5)')
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)

    plt.tight_layout()
    fig.savefig('prev_mask_dataaug.png', bbox_inches='tight')
    #plt.imsave()
    plt.show()

def func3():
    label_path = davis.label_path('bear', 4)
    label = davis.read_label(label_path)


    e_mask = morphology.erosion(label, np.ones([5, 5]))
    d_mask = morphology.dilation(label, np.ones([5, 5]))

    ax = plt.subplot(1,2,1)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    im = ax.imshow(np.uint8(label*255))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)


    ax = plt.subplot(1,2,2)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    dist_label = inputhelper.label_to_dist(label)
    im = ax.imshow(np.uint8(dist_label*255))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig('variant_prev_mask.png', bbox_inches='tight')
    #plt.imsave()
    plt.show()

def func2():
    i1_path = davis.image_path('parkour', 0)
    l1_path = davis.label_path('parkour', 0)
    l1 = davis.read_label(l1_path)
    i1 = davis.read_image(i1_path)


    i2_path = davis.image_path('drift-straight', 34)
    l2_path = davis.label_path('drift-straight', 34)
    l2 = davis.read_label(l2_path)
    i2 = davis.read_image(i2_path)

    zeros1 = np.where(l1==0)
    ones1 = np.where(l1==1)

    zeros2 = np.where(l2==0)
    ones2 = np.where(l2==1)

    cntz1 = len(zeros1[0])
    cnto1 = len(ones1[0])
    print("z:{} o:{}".format(cntz1,cnto1))

    print(len(zeros1[0])/len(ones1[0]))
    print(len(zeros2[0])/len(ones2[0]))


    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(i1)
    axes[0,1].imshow(l1)
    axes[0,0].get_xaxis().set_visible(False)
    axes[0,0].get_yaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(True)
    axes[0,1].get_yaxis().set_visible(False)
    axes[0,1].set_xticklabels([])

    axes[0,1].set_xlabel('bg/fg = {0:0.1f}'.format(len(zeros1[0])/len(ones1[0])))

    axes[1,0].imshow(i2)
    axes[1,1].imshow(l2)
    axes[1,0].get_xaxis().set_visible(False)
    axes[1,0].get_yaxis().set_visible(False)
    axes[1,1].get_xaxis().set_visible(True)
    axes[1,1].get_yaxis().set_visible(False)
    axes[1,1].set_xticklabels([])

    axes[1,1].set_xlabel('bg/fg = {0:0.1f}'.format(len(zeros2[0])/len(ones2[0])))

    plt.tight_layout()
    fig.savefig('bg_fg_ratio.png', bbox_inches='tight')
    #plt.imsave()
    plt.show()

if __name__ == '__main__':
    func1()
