import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Set model weights
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

# tf Graph input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = W * X

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

# Initializing the variables
init = tf.initialize_all_variables()

# For graphs
W_val = []
cost_val = []

# Launch the graph
sess = tf.Session()
sess.run(init)

for step in xrange(500):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

