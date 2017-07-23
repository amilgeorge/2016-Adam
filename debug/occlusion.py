'''
Created on Jan 19, 2017

@author: george
'''
from matplotlib import pyplot as plt
from skimage import morphology,io
import numpy as np


def func2():

    img1 = io.imread('/usr/stud/george/Downloads/occlusion/00057.jpg')
    img2 = io.imread('/usr/stud/george/Downloads/occlusion/00070.jpg')
    img3 = io.imread('/usr/stud/george/Downloads/occlusion/00082.jpg')








    fig, axes = plt.subplots(1,3)
    axes[0].imshow(img1)
    axes[0].get_xaxis().set_visible(True)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_xticklabels([])
    axes[0].set_xlabel('frame 57')

    axes[1].imshow(img2)
    axes[1].get_xaxis().set_visible(True)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_xticklabels([])
    axes[1].set_xlabel('frame 70')



    axes[2].imshow(img3)
    axes[2].get_xaxis().set_visible(True)
    axes[2].get_yaxis().set_visible(False)
    axes[2].set_xticklabels([])
    axes[2].set_xlabel('frame 82')

    plt.tight_layout()
    fig.savefig('out_of_view.png', bbox_inches='tight')
    #plt.imsave()
    plt.show()

if __name__ == '__main__':
    func2()
