from conv_net import ConvNet
from utils.data_utils import load_all_image_paths_convnet, load_batch_of_data_convnet
from utils.global_config import EPOCHS, BATCH_SIZE

import os
from tqdm import trange
import tensorflow as tf
import logging
import argparse
import random
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def train_model(train_data_dir: str, val_data_dir: str, save_model_dir: str):
    """

    Args:
        data_dir: Where the dataset is

    Returns:

    """
    all_train_image_paths = load_all_image_paths_convnet(train_data_dir)
    all_val_image_paths = load_all_image_paths_convnet(val_data_dir)
    log.info("All image paths loaded...")
    model = ConvNet()
    log.info("Model built...")
    BEST_VAL_LOSS = sys.maxsize

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPOCHS):
            validation_loss_epoch = 0.0

            random.shuffle(all_train_image_paths)
            full_epoch_train = trange(0, len(all_train_image_paths), BATCH_SIZE)

            for step in full_epoch_train:
                train_image_batch, train_label_batch = load_batch_of_data_convnet(
                    all_train_image_paths[step : step + BATCH_SIZE]
                )
                feed_dict_train = {
                    model.input_images: train_image_batch,
                    model.labels: train_label_batch,
                    model.dropout_prob: 0.5,
                }
                _, train_loss = sess.run(
                    [model.train_step, model.loss_fun], feed_dict_train
                )
                full_epoch_train.set_description(
                    f"Current train step loss: %g" % train_loss
                )

            for step in range(0, len(all_val_image_paths), BATCH_SIZE):
                val_image_batch, val_label_batch = load_batch_of_data_convnet(
                    all_train_image_paths[step : step + BATCH_SIZE]
                )
                feed_dict_val = {
                    model.input_images: val_image_batch,
                    model.labels: val_label_batch,
                    model.dropout_prob: 1.0,
                }
                val_loss = sess.run(model.loss_fun, feed_dict_val)
                validation_loss_epoch += val_loss

            print(f"The validation loss for epoch {e+1} is: {validation_loss_epoch}")
            if validation_loss_epoch < BEST_VAL_LOSS:
                print("Found new best! Saving model")
                saver.save(sess, f"{save_model_dir}/conv_net")
                BEST_VAL_LOSS = validation_loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script takes a directory where the train data is"
        "as well where the validation data is"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        help="Location where the data is",
        default="../data/train_data_conv",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        help="Location where the data is",
        default="../data/val_data_conv",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        help="Location where the model should be saved",
        default="../log",
    )

    args = parser.parse_args()
    train_model(args.train_data_dir, args.val_data_dir, args.save_model_dir)
