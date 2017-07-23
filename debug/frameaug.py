from dataprovider.davis_cached_2016 import DataAccessHelper
from matplotlib import pyplot as plt
from skimage import morphology
import numpy as np
import skimage
from dataprovider.transformer_rand import ImageRandomTransformer

davis = DataAccessHelper()


def _get_transform_config2():
    config = {}
    config[ImageRandomTransformer.CONFIG_ROTATION_RANGE] = [-10, 10]
    config[ImageRandomTransformer.CONFIG_ROTATION_ANGLE_STEP] = None
    config[ImageRandomTransformer.CONFIG_SHEAR_RANGE] = [-5, 5]
    config[ImageRandomTransformer.CONFIG_SHEAR_ANGLE_STEP] = None
    config[ImageRandomTransformer.CONFIG_SCALE_FACTOR_RANGE] = [1.0, 1.5]
    config[ImageRandomTransformer.CONFIG_SCALE_FACTOR_STEP] = 0.1
    config[ImageRandomTransformer.CONFIG_TRANSLATION_FACTOR_STEP] = 0.1
    config[ImageRandomTransformer.CONFIG_TRANSLATION_FACTOR_RANGE] = [-0.2, 0.2]

    return config

if __name__ == '__main__':

    image_path = davis.image_path('bear', 4)
    image = davis.read_image(image_path)
    transformer = ImageRandomTransformer(_get_transform_config2())

    image = skimage.img_as_float(image)
    fig, axes = plt.subplots(2, 3)
    axes[0,0].imshow(image)
    axes[0,0].get_xaxis().set_visible(False)
    axes[0,0].get_yaxis().set_visible(False)
    axes[0,0].set_title('Original frame')

    for i in range(1,6):
        row = int(i/3);
        col = int(i%3)
        image_trans,_ = transformer.get_random_transformed(image)
        axes[row,col].imshow(np.uint8(image_trans*255))
        axes[row,col].get_xaxis().set_visible(False)
        axes[row,col].get_yaxis().set_visible(False)

    plt.tight_layout()
    fig.savefig('dataaug.png', bbox_inches='tight')

    plt.show()