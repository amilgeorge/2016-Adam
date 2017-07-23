from dataprovider.davis_cached import DataAccessHelper


davis = DataAccessHelper()

trainseqs = davis.train_sequence_list()
testseqs = davis.test_sequence_list()

def print_tab(seq,set,len):
    print("\hline {} & {} & {} \\\\".format(seq, set, len))

def print_tab2(seq,set,len,numobjects,im_shape):
    print("\hline {} & {} & {} & {} & {}X{}\\\\".format(seq, set, len,numobjects,im_shape[0],im_shape[1]))

for seq in trainseqs:
    all_nums = davis.all_frames_nums(seq)
    n = davis.num_objects(seq)
    size = davis.get_orig_size(seq)
    print_tab2(seq,'train',len(all_nums),n,size)


for seq in testseqs:
    all_nums = davis.all_frames_nums(seq)
    n = davis.num_objects(seq)
    size = davis.get_orig_size(seq)
    print_tab2(seq, 'val', len(all_nums),n,size)