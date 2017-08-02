import tensorflow as tf

scalar = tf.constant ([6])
vector = tf.constant ([6, 4])
matrix = tf.constant ([ [6, 4], [7, 9] ])
tensor = tf.constant ([ [ [6, 4], [7, 9] ], [ [3, 4], [8, 4] ] ])

with tf.Session () as sess:
    output = sess.run (scalar)
    print ("Scalar: \n %s \n" %output)
    output = sess.run (vector)
    print ("Vector: \n %s \n" %output)
    output = sess.run (matrix)
    print ("Matrix: \n %s \n" %output)
    output = sess.run (tensor)
    print ("Tensor: \n %s \n" %output)