import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
import re

def print_all_tensors(reader):
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))

def get_tensor_stats(reader,tensor_name):

    tensor = reader.get_tensor(tensor_name)

    tensor_inp_depth = int(tensor.shape[2]/2)
    tensor_branch1 = tensor[:,:,:tensor_inp_depth,:]
    tensor_branch2 = tensor[:,:,(tensor_inp_depth):tensor_inp_depth*2,:]
    norm_b1 = np.linalg.norm(tensor_branch1.ravel(),ord=2)
    norm_b2 = np.linalg.norm(tensor_branch2.ravel(),ord=2)
    #print()
    #print()
    return norm_b1,norm_b2

def get_all_checkpoint_files(dir):
    index_files = []
    iters = []
    for file in os.listdir(dir):
        if file.endswith(".index"):

            m = re.match(r"(.*iters-)(.*)\.index", file)
            if m is None:
                continue
            iter_no = int(m.group(2))
            iters.append(iter_no)
            #index_files.append(os.path.splitext(file)[0])

    iters.sort()
    index_files = ["iters-{}".format(i) for i in iters]
    return index_files


if __name__ == '__main__':
    #file_name = '/usr/stud/george/workspace/adam/exp/mergeosvosnet-v1_no_BN-seqdavis2016-B1O-1-adam<1e-6>-opt-adam-<1e-4>-250iter-1/iters-95000'
    #file_name = '/usr/stud/george/workspace/adam/exp/adam-<1e-2>-250iter-1/iters-25000'

    #mdir = '/usr/stud/george/workspace/adam/exp/mergeosvosnet-L2_normed_fm_no_bn-seqdavis2016-B1O-1-adam<1e-6>-opt-adam-<1e-2>-250iter-1'
    #mdir = '/usr/stud/george/workspace/adam/exp/mergeosvosnet-v1_no_BN-seqdavis2016-B1O-1-adam<1e-6>-opt-adam-<1e-4>-250iter-1'
    #mdir = '/usr/stud/george/workspace/adam/exp/mergeosvosnet-L2_normed_fm_drop_no_bn-seqdavis2016-B1O-1-adam<1e-6>-opt-adam-<1e-4>-250iter-1'

    #tensor_name = 'merger/merge_conv1/ort_weights'

    #mdir = '/usr/stud/george/workspace/adam/exp/mergeosvosnet-v1_baseline_osvos_test-seqdavis2016-B1O-1-adam<1e-6>-opt-adam-<1e-4>-250iter-1'
    mdir = '/usr/stud/george/workspace/adam/exp/mergeosvosnet-v1_baseline_brn-seqdavis2016-B1O-1-adam<1e-6>-opt-adam-<1e-4>-250iter-1'

    tensor_name = 'conv_classifier/weights'


    all_ckpts = get_all_checkpoint_files(mdir)
    nb1_list =[]
    nb2_list =[]
    x_list = []
    for ckpt in all_ckpts:
        reader = tf.train.NewCheckpointReader(os.path.join(mdir,ckpt))
        #print_all_tensors(reader)
        nb1,nb2 = get_tensor_stats(reader,tensor_name)
        nb1_list.append(nb1)
        nb2_list.append(nb2)
        x_list.append(ckpt)

    fig, ax = plt.subplots()
    plt.plot(nb1_list,label = "branch1")
    plt.plot(nb2_list,label = "branch2")
    ax.set_xticklabels(x_list,rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mdir,'l2norm_{}_to_{}.png'.format(all_ckpts[0],all_ckpts[-1])))
    plt.show()

