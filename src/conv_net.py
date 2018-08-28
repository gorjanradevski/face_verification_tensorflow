import tensorflow as tf

def build_graph():

    with tf.name_scope('Placeholders'):

        X = tf.placeholder(shape=[None, 150, 150, 3], dtype=tf.float32, name='inputs')
        y = tf.placeholder(shape=[None, ], dtype=tf.int64, name='labels')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

    with tf.name_scope('Convolutional_Pooling_1'):

        W1 = tf.Variable(tf.random_uniform([5, 5, 3, 32], -1.0, 1.0), dtype=tf.float32, name='filter1')
        b1 = tf.Variable(tf.random_uniform([32], -1.0, 1.0), dtype=tf.float32, name='bias1')

        conv_op1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME',
                    use_cudnn_on_gpu=False, data_format='NHWC', dilations=[1, 1, 1, 1], name='conv1')

        conv_layer1 = tf.nn.relu(conv_op1 + b1, name='relu1')

        max_pool1 = tf.nn.max_pool(conv_layer1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='maxpool1')

    with tf.name_scope('Convolutional_Pooling_2'):

        W2 = tf.Variable(tf.random_uniform([5, 5, 32, 64], -1.0, 1.0), dtype=tf.float32, name='filter2')
        b2 = tf.Variable(tf.random_uniform([64], -1.0, 1.0), dtype=tf.float32, name='bias2')

        conv_op2 = tf.nn.conv2d(max_pool1, W2, strides=[1, 1, 1, 1], padding='SAME',
                    use_cudnn_on_gpu=False, data_format='NHWC', dilations=[1, 1, 1, 1], name='conv2')

        conv_layer2 = tf.nn.relu(conv_op2 + b2, name='relu2')

        max_pool2 = tf.nn.max_pool(conv_layer2, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC', name='maxpool2')

    with tf.name_scope('Fully_connected_layer_1'):

        flatten = tf.reshape(max_pool2, [-1, max_pool2.shape[1] * max_pool2.shape[2] * max_pool2.shape[3]])
        input_size = int(flatten.get_shape()[1])

        W3 = tf.Variable(tf.random_uniform([input_size, 128], -1.0, 1.0), dtype=tf.float32, name='Weights1')
        b3 = tf.Variable(tf.random_uniform([128], -1.0, 1.0), dtype=tf.float32, name='Bias1')

        full_1 = tf.nn.relu(tf.matmul(flatten, W3) + b3, name='Fully_connected_1')

    with tf.name_scope('Dropout_layer_1'):

        drop1 = tf.nn.dropout(full_1, keep_prob=keep_prob, name='Dropout1')

    with tf.name_scope('Logits_layer'):

        input_size = int(drop1.get_shape()[1])

        W5 = tf.Variable(tf.random_uniform([input_size, 2], -1.0, 1.0), dtype=tf.float32, name='Softmax_weights')
        b5 = tf.Variable(tf.random_uniform([2], -1.0, 1.0), dtype=tf.float32, name='Softmax_bias')

        logits = (tf.matmul(drop1, W5) + b5)

    with tf.name_scope('One_hot_labels'):
        one_hot_labels = tf.one_hot(y, depth=2)

    with tf.name_scope('Convert_to_probs'):
        softmax = tf.nn.softmax(logits, 1, name='Softmax')

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('Train'):
        optimizer = tf.train.AdamOptimizer(0.001)
        train_step = optimizer.minimize(loss)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(one_hot_labels, 1))
        compute_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    return X, y, keep_prob, loss, train_step, compute_accuracy