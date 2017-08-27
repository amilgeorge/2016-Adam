from PIL import Image
from dataprovider.davis_cached_2016 import DataAccessHelper
import numpy as np

davis = DataAccessHelper()

def mask_overlay(image,mask,color=(255, 0, 0)):


def func1():

    seq_ = ['dance-twirl',0]
    image_path = davis.image_path(seq_[0],seq_[1])
    image = davis.read_image(image_path)
    label_path = davis.label_path(seq_[0],seq_[1])
    label = davis.read_label(label_path)

    print(np.unique(label))
    pil_img = Image.fromarray(image,mode='RGB')
    red = Image.new('RGB', pil_img.size, (255, 0, 0))
    mask = Image.fromarray(label,mode='L')
    #mask.show()

    pil_img.paste(red, (0, 0),mask)
    #pil_img.show()
    pil_img.save('dance-twirl0.jpg')


def overlay_sequence(seq):
    output_folder =
    mask_folder =

    all_frame_nums = davis.all_frames_nums(seq)
    for i in all_frame_nums:
        if i==0:
            label = davis.read_label(seq,i)
        else:
            label =
    label = np.uint8(label)*123
if __name__ == '__main__':
    func1()