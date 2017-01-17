# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import input_data

batch_size = 128
test_size = 256


# 정규분포의 표준편차가 0.01 인 범위 안에서 weight 을 random 하게 설정
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 28, 28, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,  # l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

# training data 불러오고
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# traingX, testX
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# -1 can also be used to infer the shape
# -1 은 tensorflow 에서 길이를 알아서 설정하도록 하는듯
# docs 에서 reshape 는 첫 번째 인자로 모양을 변형할 tensor(행렬)을 받는데,
# 여기서는 앞에 tensor 객체를 통해 호출했으니 -1은 생략한다는 의미인 것 같다.
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])         # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])       # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])      # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625])   # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])           # FC 625 inputs, 10 outputs (labels)

# dropout 비율
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# 모델 생성
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))

"""
# 32 filters (3x3x1)
w = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# X는 이미지 데이터, strides 에서는 일반적으로 양 맨 끝은 1로 fix, 가운데 두 개는 가로 세로 stride
# padding='SAME' activation map 을 만들 때 크기를 그대로
l = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
"""