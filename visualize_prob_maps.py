import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np


if __name__ == '__main__':
    img_path = '../Results/segnetvggwithskip-res2-ch7-aug-wl-osvos-O10-1-O10-1/prob_maps/blackswan/00001.npy'
    img = np.load(img_path)
    fig = plt.figure()
    plt.imshow(img)
    plt.colorbar()
    fig.savefig('temp.png',dpi = fig.dpi)