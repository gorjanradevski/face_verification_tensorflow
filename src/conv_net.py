import tensorflow as tf
import os
from utils.global_config import IMG_WIDTH, IMG_HEIGHT, LR

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ConvNet:
    def __init__(self):
        self._create_placeholders()
        conv1 = self._conv2dmaxpool(self.input_images, 3, 32)
        conv2 = self._conv2dmaxpool(conv1, 32, 64)
        conv3 = self._conv2dmaxpool(conv2, 64, 128)
        fully1 = self._fullyconnected(conv3, 128)
        droped_output = self._create_dropout(fully1, self.is_training)
        logits = self._create_logits(droped_output)
        self._create_loss(logits, self.labels)
        self._create_optimizer()
        self._create_inference(logits)

    def _create_placeholders(self):
        self.input_images = tf.placeholder(
            shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], dtype=tf.float32, name="inputs"
        )
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="labels")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

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
            name="weights",
        )
        b = tf.Variable(
            tf.random_uniform([num_neurons], -1.0, 1.0), dtype=tf.float32, name="bias"
        )
        return tf.nn.relu(tf.matmul(flatten, W) + b, name="relu")

    def _create_dropout(self, input, is_training):
        return tf.layers.dropout(
            input,
            rate=0.5,
            noise_shape=None,
            seed=None,
            training=is_training,
            name="dropout",
        )

    def _create_logits(self, input):
        input_size = int(input.get_shape()[1])

        W = tf.Variable(tf.random_uniform([input_size, 1], -1.0, 1.0), dtype=tf.float32)
        b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype=tf.float32)

        logits = tf.matmul(input, W) + b

        return logits

    def _create_loss(self, logits, labels):
        sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        self.loss_fun = tf.reduce_mean(sigmoid_cross_entropy)

    def _create_optimizer(self):
        optimizer = tf.train.AdamOptimizer(LR)
        self.train_step = optimizer.minimize(self.loss_fun)

    def _create_inference(self, logits):
        self.prediction = tf.sigmoid(logits)

    def restore_from_checkpoint(
        self, sess: tf.Session, loader: tf.train.Saver, path_to_checkpoint_dir: str
    ):
        loader.restore(sess, path_to_checkpoint_dir)
