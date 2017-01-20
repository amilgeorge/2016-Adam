'''
Created on Jan 19, 2017

@author: george
'''
from dataprovider.davis import DataAccessHelper
from utils.helpers import save_image

davis = DataAccessHelper()

def test_sequence(sequence_name,out_dir):
    
    frames = davis.all_frames_nums(sequence_name)
    label_path = davis.label_path(sequence_name, min(frames))
    save_image(out_dir, sequence_name, min(frames), davis.read_label(label_path))

    for frame_no in range(min(frames)+1,max(frames)+1):        
        label_path = davis.label_path(sequence_name, (frame_no))
        save_image(out_dir, sequence_name, frame_no, davis.read_label(label_path))

        
    

def test_sequences():
    test_sequences = davis.test_sequence_list()+davis.train_sequence_list()
    out_dir = "../Results/dummy_mask/480p"

    for seq in test_sequences:
        test_sequence(seq, out_dir)
        

    

if __name__ == '__main__':
    test_sequences()