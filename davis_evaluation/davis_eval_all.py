import os
import argparse

import numpy   as np
import subprocess
from plot_train_graph import prepare_and_plot
import re


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

def evaluate(result_dir):
    print("Evaluating : ",result_dir)
    result_file = os.path.join(result_dir,'480p.h5')
    if os.path.exists(result_file) == True:
        return

    arg1 = os.path.join(result_dir,'480p')
    args =['python','tools/eval.py','--metrics=J',arg1,result_dir]
    print("Running with args ",args)
    p = subprocess.Popen(args,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p

def eval_all(directory):
    all_dirs = os.listdir(directory)
    processes = []
    for dir in all_dirs:
        print(dir)
        p = evaluate(os.path.join(directory,dir))
        processes.append(p)

    exit_codes = [p.wait() for p in processes if p is not None]
    iters =[]
    for d in all_dirs:
      
        m = re.match(r"(.*iter-)(.*)", d)
        if m is None:
            continue
        iter_no = int(m.group(2))
        iters.append(iter_no)

    iters.sort()
    print(iters)
    prepare_and_plot(directory,iters)





if __name__ == '__main__':
    args = parse_args()
    args.input = os.path.abspath(args.input)

    eval_all(args.input)



