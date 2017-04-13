import os
import argparse

import numpy   as np
import subprocess


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
    subprocess.Popen(args)

def eval_all(directory):
    all_dirs = os.listdir(directory)
    for dir in all_dirs:
        print(dir)
        evaluate(os.path.join(directory,dir))


if __name__ == '__main__':
    args = parse_args()
    args.input = os.path.abspath(args.input)

    eval_all(args.input)



