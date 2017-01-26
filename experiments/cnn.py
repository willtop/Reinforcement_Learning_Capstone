import tensorflow as tf


def cnn(height, width):
    x = tf.placeholder(tf.float32, [None, height, width, 3])
    y = tf.placeholder(tf.float32, [None, 2])

    weights = {
        'c1': tf.Variable(tf.random_normal([5, 5, 3, 128]), name='wc1'),
        'c2': tf.Variable(tf.random_normal([3, 3, 128, 128]), name='wc2'),
        'fc1': tf.Variable(tf.random_normal([24 * 16 * 128, 256]), name='wfc2'),
        'fc2': tf.Variable(tf.random_normal([256, 2]), name='wfc2'),
    }

    biases = {
        'c1': tf.Variable(tf.random_normal([weights['c1'].get_shape().as_list()[3]]), name='bc1'),
        'c2': tf.Variable(tf.random_normal([weights['c2'].get_shape().as_list()[3]]), name='bc2'),
        'fc1': tf.Variable(tf.random_normal([weights['fc1'].get_shape().as_list()[1]]), name='bfc1'),
        'fc2': tf.Variable(tf.random_normal([weights['fc2'].get_shape().as_list()[1]]), name='bfc2'),
    }

    # Reshape input
    x_in = tf.reshape(x, shape=[-1, height, width, 3])

    # Convolution Layers
    c1 = tf.nn.conv2d(x_in, weights['c1'], strides=[1, 2, 2, 1], padding='SAME')
    c1 = tf.nn.bias_add(c1, biases['c1'])
    c1 = tf.nn.relu(c1)

    c2 = tf.nn.conv2d(c1, weights['c2'], strides=[1, 2, 2, 1], padding='SAME')
    c2 = tf.nn.bias_add(c2, biases['c2'])
    c2 = tf.nn.relu(c2)

    # Fully connected layers
    fc1 = tf.reshape(c2, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
    return x, y, weights, biases, out