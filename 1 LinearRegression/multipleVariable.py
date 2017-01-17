import tensorflow as tf

x_data = [[1, 1, 1, 1, 1],  # bias
          [1., 0., 3., 0., 5.],  # W1
          [0., 2., 0., 4., 0.]]  # W2
y_data = [1, 2, 3, 4, 5]

"""
import numpy as np
xy = np.loadtxt("train.txt", unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]
"""

# try to find values for w and b that compute y_data = W * x1_data + b
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

# my hypothesis
hypothesis = tf.matmul(W, x_data)  # matrix multiply

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initialize the variables
init = tf.initialize_all_variables()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W)
