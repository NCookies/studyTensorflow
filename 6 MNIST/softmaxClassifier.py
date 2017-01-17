# -*- coding: utf-8 -*-

import tensorflow as tf
import input_data

# input data setting
x = tf.placeholder("float32", [None, 784])
y = tf.placeholder("float32", [None, 10])

# weight, bias 초기화
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax 모델 사용
learning_rate = 0.01
hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

# 학습을 총 몇 번 시킬 것인가
training_epoch = 25
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
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
            # cost 의 평균을 구함, 따로 나눈 것을 더한거나, 다 더해서 한 번에 나누거나...
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost / total_batch))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
