# _*_ coding: utf-8 _*_

import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import os

from Project_1 import Data
ob = Data()
ob.preDeal()

# add layer
def add_layer(input, in_size, out_size, activation_function = None):

    # original weights
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # w*x+b
    W_mul_x_plus_b = tf.matmul(input, Weights) + biases
    # activating function
    if activation_function is None:
        output = W_mul_x_plus_b
    else:
        output = activation_function(W_mul_x_plus_b)

    return output

# 3 layers net (1,10,1)
# input
x_data = ob.X #np.linspace(-1, 1, 300)[:, np.newaxis]
# noise
#noise = np.random.normal(0, 0.05, x_data.shape)
# out
y_data = ob.Y #np.square(x_data) + 1 + noise

# input data
xs = tf.placeholder(tf.float32, [None, 57])
# output data
ys = tf.placeholder(tf.float32, [None, 1])

# two hiden layer
hidden_layer = add_layer(xs, 57, 45, activation_function = tf.nn.relu)
hidden_layer1 = add_layer(hidden_layer, 45, 45, activation_function = tf.sigmoid)
# output layer
prediction = add_layer(hidden_layer1, 45, 1, activation_function = tf.sigmoid)

# nerual network parameters
# loss function
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))
# training process
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
# initialize
init = tf.global_variables_initializer()
# Session
sess = tf.Session()
# initializing
sess.run(init)
# begins
for i in range(3000):
    sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
    if i % 100 == 0:
        print sess.run(loss, feed_dict = {xs: x_data, ys: y_data})
temp = sess.run(prediction, feed_dict={xs: x_data})
aver = np.average(temp, axis = 0)
print aver
result = [0 if i < aver+0.05 else 1 for i in temp]
result = np.array(result)
cv = 0
for i, j in zip(result, y_data):
    r = float(i) - float(j[0])
    if int(r) == 0:
        cv = cv + 1
print (float(cv)/float(len(result)))

temp = sess.run(prediction, feed_dict = {xs: preprocessing.normalize(ob.testData)})
aver = np.average(temp, axis = 0)
result = [0 if i < aver else 1 for i in temp]
result = np.array(result)
path = os.path.abspath('.')
path = path + '/'+ 'project1_20453306'
try:
    with open(path, "wb") as file:
        for item in result:
            #print item
            file.write(str(float(item)))
            file.write('\n')
    file.close()
except Exception, e:
    print e

sess.close()
