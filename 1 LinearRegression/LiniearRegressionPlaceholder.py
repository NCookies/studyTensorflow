# -*- coding:utf-8 -*-

import tensorflow as tf

# input and target data
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data + b
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# hypothesis
hypothesis = w * x_data + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)  # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)  # 인자로 learning rate를 받음
train = optimizer.minimize(cost)

# before starting, initialize the variables
init = tf.initialize_all_variables()

# launch
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        print step, '\t', sess.run(cost), sess.run(w), sess.run(b)
