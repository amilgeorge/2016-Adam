from lstm import rnn
import tensorflow as tf
import  numpy as np
from tensorflow.python.client import  timeline

if __name__ == '__main__':
    IMG_HEIGHT = 480
    IMG_WIDTH = 854
    num_features = 450
    batch = 2
    x = tf.placeholder(tf.float32,shape=[None,IMG_HEIGHT,IMG_WIDTH,num_features],name='input_branch1')
    run_metadata = tf.RunMetadata()

    output = rnn.BiColRNNStatic(x,n_hidden=32,num_features=num_features)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess.run(init)
        for i in range(10):
            x_ = np.float32(np.random.rand(batch,IMG_HEIGHT,IMG_WIDTH,num_features))
            print(x_.dtype)
            print("generated x for iter{}".format(i))
            result = sess.run([output],
                              feed_dict={x:x_ },
                              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                              run_metadata=run_metadata)
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.ctf.json','w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
            print("fwd pass x for iter{}".format(i))
