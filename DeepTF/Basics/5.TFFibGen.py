import tensorflow as tf

user_input = input ()

a = tf.Variable (1)
b = tf.Variable (1)
c = tf.Variable (1)
update0 = tf.assign (c, b)
update1 = tf.assign (b, tf.add (b, a))
update2 = tf.assign (a, c)

with tf.Session () as sess:
    
    sess.run (tf.global_variables_initializer ())
    print (sess.run (a))
    print (sess.run (b))
    
    for _ in range (int (user_input)):
        
        sess.run (update0)
        sess.run (update1)
        sess.run (update2)
        
        # print (sess.run (a))
        print (sess.run (b))