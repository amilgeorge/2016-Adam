import tensorflow as tf
import numpy as np

n_hidden = 32
inp_dims = 128

def BiRowRNN(x,n_hidden,num_features):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # Define lstm cells with tensorflow
    # Forward direction cell

    def tf_encode_r(x):
        x_shape = tf.shape(x)
        return tf.reshape(x, [x_shape[0] * x_shape[1], x_shape[2], num_features])

    def tf_decode_r(x, num_images):
        x_shape = tf.shape(x)
        return tf.reshape(x, [num_images, tf.cast(x_shape[0] / num_images, tf.int32), x_shape[2], n_hidden * 2])

    x_shape = tf.shape(x)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)

    except Exception as e: # Old TensorFlow version only returns outputs not states
        print("Exception occured ",e)
        pass

    concat_outputs = tf.concat(outputs, 2)
    decode = tf_decode_r(concat_outputs, x_shape[0])

    return decode

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
    x = tf.placeholder(tf.float32, shape=(2, 4,4,3))
    arr = np.arange(0,96).reshape(2, 4,4,3)

    x_shape = tf.shape(x)
    er = tf_encode_r(x)
    oer = BiRNN(er)
    dr = tf_decode_r(oer,x_shape[0])
    print(dr)

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