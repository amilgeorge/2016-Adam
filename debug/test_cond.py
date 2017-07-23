import tensorflow as tf


pred = tf.placeholder(tf.float32, shape=[])
phase = tf.placeholder(tf.bool, shape=[])

x = tf.Variable(tf.zeros((2,3,3,2)))
k = tf.Variable([1,1,1,2])

shape =[1,1,1,2]


def update_x_1():
    return tf.identity(tf.concat([tf.ones_like(x), tf.ones_like(x) ], axis=3))


def update_x_2():
    return tf.identity(tf.concat([tf.zeros_like(x),tf.ones_like(x)*2],axis=3))


def update_x_3():
    return tf.identity(tf.concat([tf.ones_like(x)*2,tf.zeros_like(x)],axis=3))


gen = tf.random_uniform([1])
gen = tf.Print(gen,[gen])

y = tf.cond(tf.logical_and(phase,gen[0] > 0.5), lambda :tf.cond(gen[0]<0.75,update_x_2,update_x_3), update_x_1)


with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  print(y.eval(feed_dict={phase: False}))
  print(y.eval(feed_dict={phase: False}))# ==> [1]
  print(y.eval(feed_dict={phase: True}))   # ==> [2]