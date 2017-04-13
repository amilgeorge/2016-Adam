
import os
import time
import argparse

import numpy   as np
import os.path as osp

from prettytable import PrettyTable as ptable
from davis.dataset import *
from davis import log
import matplotlib.pyplot as plt


db_sequences = db_read_sequences()

test_indexes = [i for i, seq in enumerate(db_sequences) if seq.set == 'test']
train_indexes = [i for i, seq in enumerate(db_sequences) if seq.set == 'training']


def parse_args():
	"""Parse input arguments."""

	parser = argparse.ArgumentParser(
			description="""Evaluate and store results.
			""")

	parser.add_argument(
			dest='input',default=None,type=str,
			help='Path to the technique to be evaluated')

	args = parser.parse_args()

	return args

def get_data(technique,db_eval_dict):
    X = []
    for key,values in db_eval_dict[technique].iteritems():
        X.append(db_eval_dict[technique][key].values())

    X = np.hstack(X)[:,:7]
    #print X.shape
    # Adding Averages

    return technique, X

def get_plot_data_for_iter(result_file):


    if os.path.exists(result_file) == False:
        return None


    j_M_idx = 0
    j_O_idx = 1
    j_D_idx = 2
    f_M_idx = 3
    f_O_idx = 4
    f_D_idx = 5
    t_M_idx = 6

    inp = result_file
    technique = osp.splitext(osp.basename(inp))[0]

    db_eval_dict = db_read_eval(technique, raw_eval=False,
                                inputdir=osp.dirname(inp))
    t, d = get_data(technique, db_eval_dict)
    train_avgs = np.nanmean(d[train_indexes,:], axis=0)
    test_avgs = np.nanmean(d[test_indexes,:], axis=0)

    train_j_M =train_avgs[j_M_idx]
    test_j_M =test_avgs[j_M_idx]

    train_j_D = train_avgs[j_D_idx]
    test_j_D = test_avgs[j_D_idx]

    return train_j_M,test_j_M,train_j_D,test_j_D

def prepare_plot_data(test_out_dir):
    x_values = []
    train_j_M_values = []
    test_j_M_values = []
    train_j_D_values = []
    test_j_D_values = []

    for i in range(1,46):
        iter_no = i*1000
        result_file = osp.join(test_out_dir, "iter-{}".format(iter_no), '480p.h5')
        print("preparing data for iter :",iter_no)
        plot_data  = get_plot_data_for_iter(result_file)
        if plot_data == None:
            continue

        train_j_M, test_j_M, train_j_D, test_j_D = plot_data[0],plot_data[1],plot_data[2],plot_data[3]
        train_j_M_values.append(train_j_M)
        test_j_M_values.append(test_j_M)
        train_j_D_values.append(train_j_D)
        test_j_D_values.append(test_j_D)
        x_values.append(iter_no)

    return  x_values,train_j_M_values,test_j_M_values,train_j_D_values,test_j_D_values

def plot_data(x_values,train_j_M_values,test_j_M_values,train_j_D_values,test_j_D_values,save_loc):
    print("Plotting ...")
    fig = plt.figure()
    ax = plt.axes()
    ax.yaxis.grid(True)
    plt.plot(x_values,train_j_M_values,label = 'train J(M)')
    plt.plot(x_values,test_j_M_values,label = 'test J(M)')
    plt.plot(x_values,train_j_D_values,label = 'train J(D)')
    plt.plot(x_values,test_j_D_values,label = 'test J(D)')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_loc)

if __name__ == '__main__':
    args = parse_args()
    args.input = osp.abspath(args.input)
    fig_path = osp.join(args.input,'plot.png')
    x_values, train_j_M_values, test_j_M_values, train_j_D_values, test_j_D_values = prepare_plot_data(args.input)
    plot_data(x_values, train_j_M_values, test_j_M_values, train_j_D_values, test_j_D_values,fig_path)
