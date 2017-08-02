# tensorflow defines a series of operations (ops) called a graph.
# we execute them using a session which launches the graph and 
# executes them on a device

import tensorflow as tf

# build the graph
# source operations/ source ops
a = tf.constant (4)
b = tf.constant (6)

c = tf.add (a, b)

session = tf.Session ()
output = session.run (c)
session.close ()

print (output)

# we have to close the session everytime to free up memory 
# to autodelete session we could,

x1 = tf.constant ([[1, 2]])
x2 = tf.constant ([[4], [6]])

x3 = tf.matmul (x1, x2)

with tf.Session () as session:
    output_mat = session.run (x3)
    
print (output_mat)
print (type (output_mat))