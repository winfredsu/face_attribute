import tensorflow as tf
import tensorflow.contrib.slim as slim

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    w = tf.Variable(initial)
    return w


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d_strided(x, W, s, p):
    conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=p)
    return conv


def depthwise_conv2d_strided(x, W, s, p):
    conv = tf.nn.depthwise_conv2d(x, W, strides=[1, s, s, 1], padding=p)
    return conv


def conv2d_dp(input, filter_size, filter_num, stride, scope):
    channel = input.get_shape().as_list()[-1]
    filter_shape = filter_size + [channel, 1]
    with tf.variable_scope(scope):
        with tf.variable_scope('depthwise'):
            w_dw = weight_variable(filter_shape)
            # b_dw = bias_variable([channel])
            h_dw = depthwise_conv2d_strided(input, w_dw, 1, 'VALID')
            # 在训练的时候is_traning True，model frozen的时候为False
            h_dw = tf.nn.relu6(slim.batch_norm(h_dw, is_training=False))

        with tf.variable_scope('pointwise'):
            w_pw = weight_variable([1, 1, channel, filter_num])
            # b_pw = bias_variable([filter_num])
            h_pw = conv2d_strided(h_dw, w_pw, stride, 'SAME')
            h_pw = tf.nn.relu6(slim.batch_norm(h_pw, is_training=False))

        return h_pw


x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name='images')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.variable_scope('base_net'):
    x_image = x

    # first convolutional layer
    h_conv1 = conv2d_dp(input=x_image, filter_size=[3, 3], filter_num=24, stride=1, scope='conv1')
    h_conv2 = conv2d_dp(input=h_conv1, filter_size=[3, 3], filter_num=36, stride=2, scope='conv2')

    # second convolutional layer
    h_conv3 = conv2d_dp(input=h_conv2, filter_size=[3, 3], filter_num=36, stride=1, scope='conv3')
    h_conv4 = conv2d_dp(input=h_conv3, filter_size=[3, 3], filter_num=48, stride=2, scope='conv4')

    # third convolutional layer
    h_conv5 = conv2d_dp(input=h_conv4, filter_size=[3, 3], filter_num=48, stride=1, scope='conv5')
    h_conv6 = conv2d_dp(input=h_conv5, filter_size=[3, 3], filter_num=64, stride=2, scope='conv6')

    # fourth convolutional layer
    h_conv7 = conv2d_dp(input=h_conv6, filter_size=[3, 3], filter_num=64, stride=1, scope='conv7')
    h_conv8 = conv2d_dp(input=h_conv7, filter_size=[3, 3], filter_num=64, stride=1, scope='conv8')

    # FCL 1
    with tf.variable_scope('fc1'):
        W_fc1 = weight_variable([1152, 100])
        b_fc1 = bias_variable([100])
        h_fc1_flat = tf.reshape(h_conv8, [-1, 1152])
        h_fc1 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FCL 2
    with tf.variable_scope('fc2'):
        W_fc2 = weight_variable([100, 50])
        b_fc2 = bias_variable([50])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # FCL 3
    with tf.variable_scope('fc3'):
        W_fc3 = weight_variable([50, 10])
        b_fc3 = bias_variable([10])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# Output
with tf.variable_scope('output'):
    W_fc5 = weight_variable([10, 1])
    b_fc5 = bias_variable([1])
    y = tf.multiply(tf.atan(tf.matmul(h_fc3_drop, W_fc5) + b_fc5), 2, name='angle') #scale the atan output
