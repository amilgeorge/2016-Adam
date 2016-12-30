'''
Created on Dec 21, 2016

@author: george
'''
import dataprovider.davis as davis

import numpy as np

resize = [224,224]
if __name__ == '__main__':
    davis = davis.DataAccessHelper()
    train_seqs = davis.test_sequence_list()
    sum_zeros = 0
    sum_ones = 0
    for seq in train_seqs:
        all_frame_nums = davis.all_frames_nums(seq)
        for num in all_frame_nums:
            print("Seq : {} frame {}",seq,num)
            image_file = davis.image_path(seq, num)
            label_file = davis.label_path(seq, num)
            label = davis.read_label(label_file,resize).astype(np.uint8)
            ones = len(np.where(label==1)[0])
            zeros = len(np.where(label==0)[0])
            sum_zeros = sum_zeros + zeros
            sum_ones = sum_ones+ ones
    
    print("sum_ones: {} zeros{] ",sum_ones,sum_zeros)
        