import tensorflow as tf
import os
from utils.global_config import IMG_WIDTH, IMG_HEIGHT, LR

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class SiameseNet:
    def __init__(self):
        self._create_placeholders()
        net1_conv_layer1 = self._conv_layer(
            self.input_images1, [5, 5], 32, "conv_layer1", reuse=False
        )
        net2_conv_layer1 = self._conv_layer(
            self.input_images2, [5, 5], 32, "conv_layer1", reuse=True
        )
        net1_max_pooled1 = self._maxpool2d_layer(net1_conv_layer1)
        net2_max_pooled1 = self._maxpool2d_layer(net2_conv_layer1)
        net1_conv_layer2 = self._conv_layer(
            net1_max_pooled1, [5, 5], 64, "conv_layer2", reuse=False
        )
        net2_conv_layer2 = self._conv_layer(
            net2_max_pooled1, [5, 5], 64, "conv_layer2", reuse=True
        )
        net1_max_pooled2 = self._maxpool2d_layer(net1_conv_layer2)
        net2_max_pooled2 = self._maxpool2d_layer(net2_conv_layer2)
        net1_dense1 = self._dense_layer(
            net1_max_pooled2, 128, name="dense1", reuse=False, activation=tf.nn.relu()
        )
        net2_dense1 = self._dense_layer(
            net2_max_pooled2, 128, name="dense1", reuse=True, activation=tf.nn.relu()
        )
        net1_dropped = self._create_dropout(net1_dense1, self.dropout_prob)
        net2_dropped = self._create_dropout(net2_dense1, self.dropout_prob)
        net1_logits = self._dense_layer(net1_dropped, 2, name="logits", reuse=False)
        net2_logits = self._dense_layer(net2_dropped, 2, name="logits", reuse=True)

        self._create_loss(net1_logits, net2_logits, self.labels)
        self._create_optimizer()

    def _create_placeholders(self):
        self.input_images1 = tf.placeholder(
            shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], dtype=tf.float32, name="inputs1"
        )
        self.input_images2 = tf.placeholder(
            shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], dtype=tf.float32, name="inputs2"
        )
        self.labels = tf.placeholder(shape=[None], dtype=tf.int64, name="labels")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

    def _conv_layer(self, input, k_size, output_channels, name, reuse=True):
        return tf.layers.conv2d(
            input,
            output_channels,
            k_size,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=name,
            reuse=reuse,
        )

    def _maxpool2d_layer(self, input):
        return tf.layers.max_pooling2d(
            input,
            [2, 2],
            [2, 2],
            padding="valid",
            data_format="channels_last",
            name=None,
        )

    def _dense_layer(self, input, num_neurons, name, reuse=True, activation=None):
        return tf.layers.dense(
            input,
            num_neurons,
            activation=activation,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=name,
            reuse=reuse,
        )

    def _create_dropout(self, input, drop_prob):
        return tf.nn.dropout(input, keep_prob=drop_prob)

    def _create_loss(self, images1, images2, labels):
        normalized_img1 = tf.nn.l2_normalize(
            images1, axis=None, epsilon=1e-12, name=None, dim=None
        )
        normalized_img2 = tf.nn.l2_normalize(
            images2, axis=None, epsilon=1e-12, name=None, dim=None
        )
        self.loss_fun = tf.contrib.losses.metric_learning.contrastive_loss(
            labels,
            embeddings_anchor=normalized_img1,
            embeddings_positive=normalized_img2,
            margin=1.0,
        )

    def _create_optimizer(self):
        optimizer = tf.train.AdamOptimizer(LR)
        self.train_step = optimizer.minimize(self.loss_fun)
