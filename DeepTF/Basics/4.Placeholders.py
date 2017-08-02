import tensorflow as tf

a = tf.placeholder (tf.float16)
b = tf.placeholder (tf.float16)
c = b + a

dictionary = {a: [ [1, 2], [3, 4] ], b: [ [5, 6], [7, 8] ]}

with tf.Session () as sess:
    
    result = sess.run (c, feed_dict = {a: 3.5, b: 4.5})
    print (result)
    
    result = sess.run (c, feed_dict = dictionary)
    print (result)