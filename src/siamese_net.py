import tensorflow as tf
import os
from utils.global_config import IMG_WIDTH, IMG_HEIGHT, LR, CONTRASTIVE_MARGIN

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class SiameseNet:
    def __init__(self):
        self._create_placeholders()

        # Building the first network
        net1_conv_layer1 = self._conv_layer(
            self.input_images1, [5, 5], 32, "conv_layer1", reuse=False
        )
        net1_max_pooled1 = self._maxpool2d_layer(net1_conv_layer1)
        net1_conv_layer2 = self._conv_layer(
            net1_max_pooled1, [5, 5], 64, "conv_layer2", reuse=False
        )
        net1_max_pooled2 = self._maxpool2d_layer(net1_conv_layer2)
        net1_conv_layer3 = self._conv_layer(
            net1_max_pooled2, [5, 5], 128, "conv_layer3", reuse=False
        )
        net1_max_pooled3 = self._maxpool2d_layer(net1_conv_layer3)

        net1_dense1 = self._dense_layer(
            net1_max_pooled3, 128, name="dense1", reuse=False, activation=tf.nn.relu
        )
        net1_dropped = self._create_dropout(net1_dense1, self.is_training)
        net1_logits = self._dense_layer(
            net1_dropped, 128, name="logits", activation=tf.nn.relu, reuse=False
        )

        # Building the second network
        net2_conv_layer1 = self._conv_layer(
            self.input_images2, [5, 5], 32, "conv_layer1", reuse=True
        )
        net2_max_pooled1 = self._maxpool2d_layer(net2_conv_layer1)
        net2_conv_layer2 = self._conv_layer(
            net2_max_pooled1, [5, 5], 64, "conv_layer2", reuse=True
        )
        net2_max_pooled2 = self._maxpool2d_layer(net2_conv_layer2)
        net2_conv_layer3 = self._conv_layer(
            net2_max_pooled2, [5, 5], 128, "conv_layer3", reuse=True
        )
        net2_max_pooled3 = self._maxpool2d_layer(net2_conv_layer3)

        net2_dense1 = self._dense_layer(
            net2_max_pooled3, 128, name="dense1", reuse=True, activation=tf.nn.relu
        )
        net2_dropped = self._create_dropout(net2_dense1, self.is_training)
        net2_logits = self._dense_layer(
            net2_dropped, 128, name="logits", activation=tf.nn.relu, reuse=True
        )

        # Preparing for the loss / flattening the output
        net1_flatten = tf.layers.flatten(net1_logits)
        net2_flatten = tf.layers.flatten(net2_logits)
        self._create_loss(net1_flatten, net2_flatten, self.labels)
        self._create_optimizer()
        self._inference(net1_flatten, net2_flatten)

    def _create_placeholders(self):
        self.input_images1 = tf.placeholder(
            shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], dtype=tf.float32, name="inputs1"
        )
        self.input_images2 = tf.placeholder(
            shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], dtype=tf.float32, name="inputs2"
        )
        self.labels = tf.placeholder(shape=[None], dtype=tf.int64, name="labels")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

    def _conv_layer(self, input, k_size, output_channels, name, reuse):
        return tf.layers.conv2d(
            input,
            output_channels,
            k_size,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
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

    def _dense_layer(self, input, num_neurons, name, reuse, activation=None):
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

    def _create_dropout(self, input, is_training):
        return tf.layers.dropout(
            input,
            rate=0.5,
            noise_shape=None,
            seed=None,
            training=is_training,
            name=None,
        )

    def _create_loss(self, images1, images2, labels):
        distance = tf.reduce_sum(tf.square(images1 - images2), 1)
        distance_sqrt = tf.sqrt(distance)

        loss = (
            labels * tf.square(tf.maximum(0., CONTRASTIVE_MARGIN - distance))
            + (1 - labels) * distance_sqrt
        )

        self.loss_fun = 0.5 * tf.reduce_mean(loss)

    def _create_optimizer(self):
        optimizer = tf.train.AdamOptimizer(LR)
        self.train_step = optimizer.minimize(self.loss_fun)

    def _inference(self, flat_image1, flat_image2):
        return tf.norm(flat_image1 - flat_image2, ord="euclidean")
