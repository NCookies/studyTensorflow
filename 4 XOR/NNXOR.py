import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Add histogram
w1_hist = tf.histogram_summary("weights1", W1)
w2_hist = tf.histogram_summary("weights2", W2)

b1_hist = tf.histogram_summary("biases1", b1)
b2_hist = tf.histogram_summary("biases2", b2)

y_hist = tf.histogram_summary("y", Y)

with tf.name_scope('Layer2') as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope('Layer3') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

alpha = tf.Variable(0.1)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/xor", sess.graph)

    sess.run(init)

    for step in xrange(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)
            print 'Step :', step
            print 'Cost :', sess.run(cost, feed_dict={X: x_data, Y: y_data})

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data})
    print "Accuracy :", accuracy.eval({X: x_data, Y: y_data})
