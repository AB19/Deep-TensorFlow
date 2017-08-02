# mnist is a multi-class classification problem
import tensorflow as tf

# import matplotlib for plotting
import matplotlib.pyplot as plt

# tensorflow already has the mnist data-set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets ("/tmp/data/", one_hot = True)

# define the training set and the test set
train_set = int (mnist.train.num_examples)
test_set = int (mnist.test.num_examples)

# define the number of nodes in each hidden layer
nodes_h1 = 784
nodes_h2 = 500
nodes_h3 = 10

# the size of the output layer
num_labels = 10

# size of the batch = number of inputs sent at a time
batch_size = 100

# create the placeholders
x = tf.placeholder (tf.float64, [None, 784], name = "x")
y = tf.placeholder (tf.float64, name = "y")

# now define the neural network model
def neural_net (data):
    
    hidden_1_layer = {'weights': tf.Variable (tf.random_normal ([784, nodes_h1],
                                                                dtype = tf.float64,
                                                                mean = 0,
                                                                stddev = 0.01)),
                      'biases': tf.Variable (tf.random_normal ([nodes_h1],
                                                               dtype =tf.float64,
                                                               mean = 0,
                                                               stddev = 0.01))}
                                                               
    hidden_2_layer = {'weights': tf.Variable (tf.random_normal ([nodes_h1, nodes_h2],
                                                                dtype = tf.float64,
                                                                mean = 0,
                                                                stddev = 0.01)),
                      'biases': tf.Variable (tf.random_normal ([nodes_h2],
                                                               dtype =tf.float64,
                                                               mean = 0,
                                                               stddev = 0.01))}
                                                               
    hidden_3_layer = {'weights': tf.Variable (tf.random_normal ([nodes_h2, nodes_h3],
                                                                dtype = tf.float64,
                                                                mean = 0,
                                                                stddev = 0.01)),
                      'biases': tf.Variable (tf.random_normal ([nodes_h3],
                                                               dtype =tf.float64,
                                                               mean = 0,
                                                               stddev = 0.01))} 
                                                               
    output_layer = {'weights': tf.Variable (tf.random_normal ([nodes_h3, num_labels],
                                                                dtype = tf.float64,
                                                                mean = 0,
                                                                stddev = 0.01)),
                      'biases': tf.Variable (tf.random_normal ([num_labels],
                                                               dtype =tf.float64,
                                                               mean = 0,
                                                               stddev = 0.01))}                                                           
                                                                
    # input * weights + biases
    
    l1 = tf.add (tf.matmul (data, hidden_1_layer ['weights']), hidden_1_layer ['biases'])
    l1 = tf.nn.relu (l1)
    
    l2 = tf.add (tf.matmul (l1, hidden_2_layer ['weights']), hidden_2_layer ['biases'])
    l2 = tf.nn.relu (l2)
    
    l3 = tf.add (tf.matmul (l2, hidden_3_layer ['weights']), hidden_3_layer ['biases'])
    l3 = tf.nn.relu (l3)
    
    output = tf.add (tf.matmul (l3, output_layer ['weights']), output_layer ['biases'])
    
    return output
    
def train_neural_net (x):
    
    pred = neural_net (x)
    cost = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (logits = pred, labels = y))
    learning_rate = 0.001
    # epoch = feed forward + backward prop
    epochs = 70
    optimizer = tf.train.AdamOptimizer (learning_rate).minimize (cost)
    
    init = tf.global_variables_initializer ()
    
    with tf.Session () as sess:
        
        sess.run (init)
        costs = []
        for epoch in range (epochs):
            epoch_loss = 0
            for _ in range (550):
                
                epoch_x, epoch_y = mnist.train.next_batch (batch_size)
                _, c = sess.run ([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss = epoch_loss + c
            costs.append (epoch_loss)
            print ("Epoch" + str (epoch + 1) + "-loss: " + str (epoch_loss))
            
        correct = tf.equal (tf.arg_max (pred, 1), tf.argmax (y, 1))
        
        accuracy = tf.reduce_mean (tf.cast (correct, 'float'))
        print ("Accuracy:" + str (accuracy.eval ({x: mnist.test.images, y: mnist.test.labels})))
        
        plt.plot (epoch_loss, range (70))
        plt.show ()
        
train_neural_net (x)        