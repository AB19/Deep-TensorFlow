import tensorflow as tf

counter = tf.Variable (1)

one = tf.constant (1)
# new_count = tf.add (counter, one)
update = tf.assign (counter, tf.add (counter, one))

two = tf.constant (2)
two = tf.add (one, two)

with tf.Session () as sess:
    
    sess.run (tf.global_variables_initializer ())
    print (sess.run (counter))
    # print (sess.run (one))
    for _ in range (10):
        
        # sess.run (new_count)
        sess.run (update)
        print (sess.run (counter))   
        
    try:
        print (sess.run (two))
    except:
        print ("Cannot add and replace existing constants")


