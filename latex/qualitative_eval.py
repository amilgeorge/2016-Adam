from dataprovider.davis_cached_2016 import DataAccessHelper
from shutil import copyfile
import os


davis = DataAccessHelper()

trainseqs = davis.train_sequence_list()
testseqs = davis.test_sequence_list()

def print_tab(seq,set,len):
    print("\hline {} & {} & {} \\\\".format(seq, set, len))

def print_tab2(seq,set,len,numobjects,im_shape):
    print("\hline {} & {} & {} & {} & {}X{}\\\\".format(seq, set, len,numobjects,im_shape[0],im_shape[1]))

def copy_for_seq(seq,sel_frames):
    dst_dir = '/usr/stud/george/tpsm_dist_transform_eval/'
    src_dir = '/usr/stud/george/workspace/adam/test_out/' \
              's480pvgg-segnet_brn-daviscombo-O1-Plabel_to_dist-osvosold-reg1e-4-mo<1e-2>-de-scale1.3-1/' \
              'iter-685000/480p-vis'

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for n in sel_frames:
        src_file = os.path.join(src_dir,seq,'{0:05}.jpg'.format(n))
        dst_file = os.path.join(dst_dir, get_name(seq,n))

        copyfile(src_file,dst_file)


def print_for_seq(seq,sel_frames):
    assert len(sel_frames) ==8,"8 frames not selected"
    print("\hline {} \\\\".format(seq))

    print("\hline")
    for i,no in enumerate(sel_frames):
        name = get_name(seq,no)
        end ='&'
        if i==3 or i==7:
            end = "\\\\"
        print("{{\includegraphics[width=0.25\linewidth]{{tpsm_dist_transform_eval/{0}}}}}{1}".format(name,end))

def get_name(seq,frame_no):
    return "{0}_{1:05}.jpg".format(seq,frame_no)

for seq in testseqs:
    all_nums = davis.all_frames_nums(seq)
    max_num = max(all_nums)
    num_parts = 8
    step = (max_num)/(num_parts-1)
    sel_frames = []
    sel_frames.append(0)
    for i in range(1,num_parts):
        frame_no = int(step*i)
        sel_frames.append(frame_no)
    print_for_seq(seq,sel_frames)
    copy_for_seq(seq,sel_frames)
