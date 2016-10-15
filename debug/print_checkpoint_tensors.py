'''
Created on Oct 11, 2016

@author: george
'''

import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from net.coarsenet import CoarseNet
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.learn.python.learn.utils import checkpoints

def print_tensors_in_checkpoint_file(file_name, tensor_name):
  """Prints tensors in a checkpoint file.
  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.
  If `tensor_name` is provided, prints the content of the tensor.
  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
  """
  try:
    if not tensor_name:
      variables = checkpoints.list_variables(file_name)
      for name, shape in variables:
        print("%s\t%s" % (name, str(shape)))
    else:
      print("tensor_name: ", tensor_name)
      print(checkpoints.load_variable(file_name, tensor_name))
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")


print_tensors_in_checkpoint_file('exp/coarse-weighted-f5-5/iters-11000',None)
# inp = tf.placeholder(tf.float32,shape=[None,224,224,4],name='input')
# coarse_net = CoarseNet(inp,'resnet_v1_50',False)
# net,end_points = coarse_net.net,coarse_net.end_points
# 
# 
# # Attach sigmoid and reshape
# coarse_out= tf.reshape(tf.sigmoid(net),[-1,56,56,1])
# sess = tf.InteractiveSession()
# global_step_var = tf.Variable(0, trainable=False)
# 
# coarse_net.initialize(sess,'exp/coarse-weighted-f5-5/iters-12000')
# a= tf.get_collection('moving_mean')
# 
# print("end")