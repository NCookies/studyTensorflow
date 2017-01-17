# -*- coding: utf-8 -*-

import tensorflow as tf
import input_data


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

# input data setting
x = tf.placeholder("float32", [None, 784])
y = tf.placeholder("float32", [None, 10])

# Store layers weight & bias
W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 500], initializer=xavier_init(256, 500))
W3 = tf.get_variable("W3", shape=[500, 256], initializer=xavier_init(500, 256))
W4 = tf.get_variable("W4", shape=[256, 256], initializer=xavier_init(256, 256))
W5 = tf.get_variable("W5", shape=[256, 10], initializer=xavier_init(256, 10))

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([500]))
b3 = tf.Variable(tf.random_normal([256]))
b4 = tf.Variable(tf.random_normal([256]))
b5 = tf.Variable(tf.random_normal([10]))

# ReLU 모델 사용
dropout_rate = tf.placeholder("float")

_L2 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
L2 = tf.nn.dropout(_L2, dropout_rate)

_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W2), b2))
L3 = tf.nn.dropout(_L3, dropout_rate)

_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W3), b3))
L4 = tf.nn.dropout(_L4, dropout_rate)

_L5 = tf.nn.relu(tf.add(tf.matmul(L4, W4), b4))
L5 = tf.nn.dropout(_L5, dropout_rate)

hypothesis = tf.add(tf.matmul(L5, W5), b5)

learning_rate = 0.001
# cost function 으로는 cross-entropy 함수 사용
# hypothesis 에 softmax 를 취해주지 않았기 때문에 softmax_cross_entropy_with_logits 메서드 사용
# (0과 1 사이의 값을 얻기 위해서 softmax 사용)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

# 학습을 총 몇 번 시킬 것인가
training_epoch = 100
# ????
display_step = 1
# 한 묶음에 training data 몇 개?
batch_size = 100
# training data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:
    sess.run(init)

    # 학습 몇 번 시킬거니
    for epoch in range(training_epoch):
        # 평균 에러값??
        avg_cost = 0
        # training data 를 몇 개의 묶음으로 나누어서 학습할 것인지
        total_batch = int(mnist.train.num_examples/batch_size)

        # x_1, x_2, x_3 ... , x_total_batch
        for i in range(total_batch):
            # training data 에서 batch_size 만큼 읽어옴
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 데이터를 이용하여 학습
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.5})
            # cost 의 평균을 구함, 따로 나눈 것을 더한거나, 다 더해서 한 번에 나누거나...
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.5}) / total_batch

        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate: 1}))