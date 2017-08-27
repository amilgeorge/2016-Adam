import tensorflow as tf
import numpy as np

def BiColRNNStatic(x,n_hidden,num_features,scope_name):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # Define lstm cells with tensorflow
    # Forward direction cell

    #x.set_shape((2,480,854,num_features))
    def tf_encode_c(x):
        x = tf.transpose(x,perm=[0, 2, 1, 3])
        x_shape = tf.shape(x)
        reshaped = tf.reshape(x, [x_shape[0] * x_shape[1], x_shape[2], num_features])
        reshaped = tf.Print(reshaped, [tf.shape(reshaped)], message="encoded shape")
        return reshaped

    def tf_decode_c(x, num_images):
        x_shape = tf.shape(x)

        shape_target = [num_images, tf.cast(x_shape[0] / num_images, tf.int32), x_shape[1], n_hidden * 2]
        #shape_target = tf.Print(shape_target,[x_shape,shape_target],message="shape")
        reshaped = tf.reshape(x,shape_target)
        reshaped = tf.transpose(reshaped,perm=[0, 2, 1, 3])
        return reshaped

    with tf.variable_scope(scope_name) as scope:
        x_shape = tf.shape(x)
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        inp = tf_encode_c(x)

        inp = tf.unstack(inp,num=480,axis=1)
        # Get lstm cell output
        try:
            #inp = tf.Print(inp, [inp], message="before bidirectional")

            outputs, _,_ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp,
                                                         dtype=tf.float32)
            #outputs = tf.Print(outputs,[outputs],message="after bidirectional")

        except Exception as e: # Old TensorFlow version only returns outputs not states
            print("Exception occured ",e)
            pass

        concat_outputs = tf.stack(outputs,axis=1)
        concat_outputs = tf.Print(concat_outputs,[tf.shape(concat_outputs),x_shape])
        #print(concat_outputs)
        decode = tf_decode_c(concat_outputs, x_shape[0])
        decode = tf.Print(decode,[decode],message="decoded c")
        return decode

def BiRowRNNStatic(x,n_hidden,num_features,scope_name):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # Define lstm cells with tensorflow
    # Forward direction cell

    #x.set_shape((2,480,854,num_features))
    def tf_encode_r(x):
        x_shape = tf.shape(x)
        reshaped = tf.reshape(x, [x_shape[0] * x_shape[1], x_shape[2], num_features])
        reshaped = tf.Print(reshaped, [tf.shape(reshaped)], message="encoded shape")
        return reshaped

    def tf_decode_r(x, num_images):
        x_shape = tf.shape(x)

        shape_target = [num_images, tf.cast(x_shape[0] / num_images, tf.int32), x_shape[1], n_hidden * 2]
        #shape_target = tf.Print(shape_target,[x_shape,shape_target],message="shape")
        reshaped = tf.reshape(x,shape_target)
        return reshaped

    with tf.variable_scope(scope_name) as scope:
        x_shape = tf.shape(x)
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        inp = tf_encode_r(x)

        inp = tf.unstack(inp,num=854,axis=1)
        # Get lstm cell output
        try:
            #inp = tf.Print(inp, [inp], message="before bidirectional")

            outputs, _,_ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp,
                                                         dtype=tf.float32)
            #outputs = tf.Print(outputs,[outputs],message="after bidirectional")

        except Exception as e: # Old TensorFlow version only returns outputs not states
            print("Exception occured ",e)
            pass

        concat_outputs = tf.stack(outputs,axis=1)
        concat_outputs = tf.Print(concat_outputs,[tf.shape(concat_outputs),x_shape])
        #print(concat_outputs)
        decode = tf_decode_r(concat_outputs, x_shape[0])
        decode = tf.Print(decode,[decode],message="decoded")
        return decode

def BiRowRNN(x,n_hidden,num_features):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # Define lstm cells with tensorflow
    # Forward direction cell

    x.set_shape((2,480,854,num_features))
    def tf_encode_r(x):
        x_shape = tf.shape(x)
        reshaped = tf.reshape(x, [x_shape[0] * x_shape[1], x_shape[2], num_features])
        reshaped = tf.Print(reshaped, [tf.shape(reshaped)], message="encoded shape")
        return reshaped

    def tf_decode_r(x, num_images):
        x_shape = tf.shape(x)

        shape_target = [num_images, tf.cast(x_shape[0] / num_images, tf.int32), x_shape[1], n_hidden * 2]
        #shape_target = tf.Print(shape_target,[x_shape,shape_target],message="shape")
        reshaped = tf.reshape(x,shape_target)
        return reshaped

    x_shape = tf.shape(x)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    inp = tf_encode_r(x)
    # Get lstm cell output
    try:
        inp = tf.Print(inp, [inp], message="before bidirectional")
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inp,
                                              dtype=tf.float32)

        outputs, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp,
                                                     dtype=tf.float32)
        #outputs = tf.Print(outputs,[outputs],message="after bidirectional")

    except Exception as e: # Old TensorFlow version only returns outputs not states
        print("Exception occured ",e)
        pass

    concat_outputs = tf.concat(outputs, 2)
    concat_outputs = tf.Print(concat_outputs,[tf.shape(concat_outputs),x_shape])
    #print(concat_outputs)
    decode = tf_decode_r(concat_outputs, x_shape[0])
    decode = tf.Print(decode,[decode],message="decoded")
    return decode