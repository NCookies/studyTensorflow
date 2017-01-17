import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)
    if step % 100 == 0:
        # plt.plot(sess.run(tf.matmul(W, x_data)[0]), y_data, 'ro')
        # plt.plot(sess.run(hypothesis))
        plt.plot(x_data[0], x_data[1], 'ro')
        plt.plot(sess.run(tf.matmul(W, x_data)[0]), y_data)
        plt.ylabel('y')
        plt.xlabel('X')
        plt.xlim(0, 10)
        plt.ylim(-2, 2)
        plt.show()
