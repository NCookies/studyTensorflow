# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o']
# {'h': 0, 'e': 1, 'l': 2, 'o': 3}
char_dic = {w: i for i, w in enumerate(char_rdic)}
print char_dic

# [0, 1, 2, 2, 3]
ground_truth = [char_dic[c] for c in 'hello']

"""
x_data = np.array([[1, 0, 0, 0],  # h
                  [0, 1, 0, 0],  # e
                  [0, 0, 1, 0],  # l
                  [0, 0, 1, 0]],  # l
                  dtype='f')
"""
# 마지막 글자인 o를 제외하고 one-hot vector 생성
x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)

# Configuration
# output 의 크기
rnn_size = len(char_dic)  # 4
batch_size = 1
output_size = 4

# RNN Model
# <tensorflow.python.ops.rnn_cell.BasicRNNCell object at 0x7f983b459610>
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size, input_size=None,
                                       #deprecated at tensorflow 0.9
                                       #activation = tanh,
                                       )
print(rnn_cell)

# Tensor("zeros:0", shape=(1, 4), dtype=float32)
initial_state = rnn_cell.zero_state(batch_size, tf.float32)
print(initial_state)

"""
initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size])  # 위의 코드와 같은 결과
print(initial_state_1)
"""

"""
[[1,0,0,0]] # h
[[0,1,0,0]] # e
[[0,0,1,0]] # l
[[0,0,1,0]] # l
"""
# 입력 array 의 모양
# array 의 개수 => hidden layer 가 가로로 몇 개나 있는가
x_split = tf.split(0, len(char_dic), x_data)  # 가로축으로 4개로 split
print(x_split)

# outputs: [<tf.Tensor 'RNN/BasicRNNCell/Tanh:0' shape=(1, 4) dtype=float32>, <tf.Tensor 'RNN/BasicRNNCell_1/Tanh:0' shape=(1, 4) dtype=float32>, <tf.Tensor 'RNN/BasicRNNCell_2/Tanh:0' shape=(1, 4) dtype=float32>, <tf.Tensor 'RNN/BasicRNNCell_3/Tanh:0' shape=(1, 4) dtype=float32>]
# state: Tensor("RNN/BasicRNNCell_3/Tanh:0", shape=(1, 4), dtype=float32)
# initial state: 초기 weight 값을 무엇으로 줄 것인가
outputs, state = tf.nn.rnn(cell=rnn_cell, inputs=x_split, initial_state=initial_state)

logits = tf.reshape(tf.concat(1, outputs),  # shape = 1 x 16
                    [-1, rnn_size])         # shape = 4 x 4
# (4, 4)
print(logits.get_shape())
"""
[[logit from 1st output],
[logit from 2nd output],
[logit from 3rd output],
[logit from 4th output]]
"""

targets = tf.reshape(ground_truth[1:], [-1])  # a shape of [-1] flattens into 1-D
targets.get_shape()

weights = tf.ones([len(char_dic) * batch_size])

# logits : 예측값
# targets : 목표값
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print(result, [char_rdic[t] for t in result])
