import os
os.chdir ("C:/Users/abhil/Desktop/DeepTF/Linear Regression")

# import the dataset
import pandas as pd
import numpy as np
data = pd.read_csv ("Salary_Data.csv")
# extract features
data_X = data.iloc [:, : -1].values
# reshape the data to retain their structure
data_X = data_X.reshape (30, 1)
data_Y = data.iloc [:, -1].values
data_Y = data_Y.reshape (30, 1)

# Training and testing Data
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split (data_X, data_Y, test_size = 0.2)

# find the number of features in x and y
num_features = data_X.shape [1]
num_labels = data_Y.shape [1]

import tensorflow as tf
# tf Graph Input
X = tf.placeholder (tf.float64, [None, num_features], name = "X")
Y = tf.placeholder (tf.float64, [None, num_labels], name = "Y")

# Set model weights
W = tf.Variable (tf.random_normal ( [num_features, num_labels],
                                   dtype = tf.float64,
                                   mean = 0,
                                   stddev = 0.01,
                                   name = "weights"))

b = tf.Variable (tf.random_normal ( [1, num_labels],
                                   dtype = tf.float64,
                                   mean = 0,
                                   stddev = 0.01,
                                   name = "bias"))

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Construct a linear model
pred = tf.add (tf.multiply (X, W), b)

# Mean squared error
cost = tf.reduce_mean (tf.square (pred - Y))
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer (learning_rate).minimize (cost)

# Initializing the variables
init = tf.global_variables_initializer()

# cost history
cost_history = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range (training_epochs):
        for (x, y) in zip (train_X, train_Y):
            x = x.reshape (1, 1)
            y = y.reshape (1, 1)
            sess.run (optimizer, feed_dict = {X: x, Y: y})
            c = sess.run (cost, feed_dict={X: train_X, Y:train_Y})
            cost_history.append (cost)
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    
    import matplotlib.pyplot as plt
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()