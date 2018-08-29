import tensorflow as tf
from utils.global_config import IMG_WIDTH, IMG_HEIGHT, TRAIN_DROP, LR


class ConvNet:
    def __init__(self):
        self._create_placeholders()
        conv1 = self._conv2dmaxpool(self.input_image, 3, 32)
        conv2 = self._conv2dmaxpool(conv1, 32, 64)
        fully1 = self._fullyconnected(conv2, 32)
        drop = self._dropout(fully1)
        logits = self._create_logits(drop)
        self._create_loss(logits, self.label)
        self._create_optimizer()
        self._inference_module(logits)

    def _create_placeholders(self):
        self.input_image = tf.placeholder(
            shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], dtype=tf.float32, name="inputs"
        )
        self.label = tf.placeholder(shape=[None], dtype=tf.int64, name="labels")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

    def _conv2dmaxpool(self, X, in_channels, out_channels):
        W = tf.Variable(
            tf.random_uniform([5, 5, in_channels, out_channels], -1.0, 1.0),
            dtype=tf.float32,
            name="filter",
        )
        b = tf.Variable(
            tf.random_uniform([out_channels], -1.0, 1.0), dtype=tf.float32, name="bias"
        )

        conv_op = tf.nn.conv2d(
            X,
            W,
            strides=[1, 1, 1, 1],
            padding="SAME",
            use_cudnn_on_gpu=False,
            data_format="NHWC",
            dilations=[1, 1, 1, 1],
            name="conv",
        )

        conv_layer = tf.nn.relu(conv_op + b, name="relu")

        return tf.nn.max_pool(
            conv_layer,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            data_format="NHWC",
            name="maxpool",
        )

    def _fullyconnected(self, input, num_neurons):
        flatten = tf.reshape(
            input, [-1, input.shape[1] * input.shape[2] * input.shape[3]]
        )
        input_size = int(flatten.get_shape()[1])

        W = tf.Variable(
            tf.random_uniform([input_size, num_neurons], -1.0, 1.0),
            dtype=tf.float32,
            name="Weights",
        )
        b = tf.Variable(
            tf.random_uniform([num_neurons], -1.0, 1.0), dtype=tf.float32, name="Bias"
        )
        return tf.nn.relu(tf.matmul(flatten, W) + b, name="relu")

    def _dropout(self, input):
        drop_out = tf.nn.dropout(input, keep_prob=TRAIN_DROP)

        return drop_out

    def _create_logits(self, input):
        input_size = int(input.get_shape()[1])

        W = tf.Variable(tf.random_uniform([input_size, 2], -1.0, 1.0), dtype=tf.float32)
        b = tf.Variable(tf.random_uniform([2], -1.0, 1.0), dtype=tf.float32)

        logits = tf.matmul(input, W) + b

        return logits

    def _create_loss(self, logits, labels):
        one_hot_labels = tf.one_hot(labels, depth=2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels, logits=logits
        )
        self.loss = tf.reduce_mean(cross_entropy)

    def _create_optimizer(self):
        optimizer = tf.train.AdamOptimizer(LR)
        self.train_step = optimizer.minimize(self.loss)

    def _inference_module(self, logits):
        return tf.nn.softmax(logits, 1, name="Softmax")
