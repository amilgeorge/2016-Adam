#import tensorflow as tf
import numpy as np

def encode(t):
    ut = tf.unstack(t,axis=0)
    print(ut)

    return ut

def encode_r(t):
    return arr.reshape(8,4,3)
def decode_r(t,numimages):
    shape = t.shape
    return arr.reshape(numimages,int(shape[0]/numimages),shape[1],shape[2])
def encode_c(t):
    arr = t.transpose([0, 2, 1, 3])
    return arr.reshape(8, 4, 3)

def decode_c(t,numimages):
    arr = t
    shape = t.shape
    arr = arr.reshape(numimages,int(shape[0]/numimages),shape[1],shape[2])
    return arr.transpose([0,2,1,3])

if __name__ == '__main__':
 #   x = tf.placeholder(tf.float32, shape=(2, 4,4,3))
    arr = np.arange(0,96).reshape(2, 4,4,3)

    er = encode_r(arr)
    dr = decode_r(er,arr.shape[0])

    if np.array_equal(arr,dr):
        print("True")
    else:
        print("False")

    enc_c = encode_c(arr)
    dc = decode_c(enc_c,2)

    if np.array_equal(arr, dc):
        print("True")
    else:
        print("False")


    """
    
    print(arr[1,:,0,0])
    arr = arr.reshape(8,4,3)
    print(arr[4,:,0])
    x = encode(x)
"""