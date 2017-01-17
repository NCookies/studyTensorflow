import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
w2 = tf.Variable(tf.random_uniform([5, 10], -1.0, 1.0), name='weight2')
w3 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight3')
w4 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight4')
w5 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight5')
w6 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight6')
w7 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight7')
w8 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0), name='weight8')

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b3 = tf.Variable(tf.zeros([10]), name="Bias3")
b2 = tf.Variable(tf.zeros([10]), name="Bias2")
b4 = tf.Variable(tf.zeros([10]), name="Bias4")
b5 = tf.Variable(tf.zeros([10]), name="Bias5")
b6 = tf.Variable(tf.zeros([10]), name="Bias6")
b7 = tf.Variable(tf.zeros([10]), name="Bias7")
b8 = tf.Variable(tf.zeros([1]), name="Bias8")

with tf.name_scope('Layer2') as scope:
    L2 = tf.sigmoid(tf.matmul(X, w1) + b1)
with tf.name_scope('Layer3') as scope:
    L3 = tf.sigmoid(tf.matmul(L2, w2) + b2)
with tf.name_scope('Layer4') as scope:
    L4 = tf.sigmoid(tf.matmul(L3, w3) + b3)
with tf.name_scope('Layer5') as scope:
    L5 = tf.sigmoid(tf.matmul(L4, w4) + b4)
with tf.name_scope('Layer6') as scope:
    L6 = tf.sigmoid(tf.matmul(L5, w5) + b5)
with tf.name_scope('Layer7') as scope:
    L7 = tf.sigmoid(tf.matmul(L6, w6) + b6)
with tf.name_scope('Layer8') as scope:
    L8 = tf.sigmoid(tf.matmul(L7, w7) + b7)

with tf.name_scope('last') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L8, w8) + b8)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

with tf.name_scope("accuracy") as scope:
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_sum = tf.scalar_summary("accuracy", accuracy)

alpha = tf.Variable(0.1)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./logs/deep', sess.graph)

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
