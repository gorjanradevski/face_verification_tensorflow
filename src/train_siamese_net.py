from siamese_net import SiameseNet
from utils.data_utils import load_all_image_paths_siamese, load_batch_of_data_siamese
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


def train_model(pairs_train_path: str, pairs_val_path: str, save_model_dir: str):
    """

    Args:
        data_dir: Where the dataset is

    Returns:

    """
    all_train_image_paths = load_all_image_paths_siamese(pairs_train_path)
    pos_train_image_paths = all_train_image_paths[:200]
    neg_train_image_paths = all_train_image_paths[-200:]
    train_image_paths = pos_train_image_paths + neg_train_image_paths

    all_val_image_paths = load_all_image_paths_siamese(pairs_val_path)
    pos_val_image_paths = all_val_image_paths[:32]
    neg_val_image_paths = all_val_image_paths[-32:]
    val_image_paths = pos_val_image_paths + neg_val_image_paths


    log.info("All image paths loaded...")
    model = SiameseNet()
    log.info("Model built...")
    BEST_VAL_LOSS = sys.maxsize

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPOCHS):
            validation_loss_epoch = 0.0

            random.shuffle(train_image_paths)
            full_epoch_train = trange(0, len(train_image_paths), BATCH_SIZE)

            for step in full_epoch_train:
                train_image1_batch, train_image2_batch, train_label_batch = load_batch_of_data_siamese(
                    train_image_paths[step : step + BATCH_SIZE]
                )
                feed_dict_train = {
                    model.input_images1: train_image1_batch,
                    model.input_images2: train_image2_batch,
                    model.labels: train_label_batch,
                    model.is_training: True,
                }

                _, train_loss = sess.run(
                    [model.train_step, model.loss_fun], feed_dict_train
                )
                full_epoch_train.set_description(
                    f"Current train step loss: %g" % train_loss
                )

            log.info("Evaluating model on the validation set")
            for step in range(0, len(val_image_paths), BATCH_SIZE):
                val_image1_batch, val_image2_batch, val_label_batch = load_batch_of_data_siamese(
                    val_image_paths[step : step + BATCH_SIZE]
                )
                feed_dict_val = {
                    model.input_images1: val_image1_batch,
                    model.input_images2: val_image2_batch,
                    model.labels: val_label_batch,
                    model.is_training: False,
                }
                val_loss = sess.run(model.loss_fun, feed_dict_val)
                validation_loss_epoch += val_loss

            print(f"The validation loss for epoch {e+1} is: {validation_loss_epoch}")
            if validation_loss_epoch < BEST_VAL_LOSS:
                print("Found new best! Saving model...")
                saver.save(sess, f"{save_model_dir}/siamese_net")
                BEST_VAL_LOSS = validation_loss_epoch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script takes a path where the"
        "train pairs are as well as where the validation pairs are"
    )
    parser.add_argument(
        "--train_pairs_path",
        type=str,
        help="Location the train paths are",
        default="../data/train_data_siamese/pairsDevTrain.txt",
    )
    parser.add_argument(
        "--val_pairs_path",
        type=str,
        help="Location where the validation pairs are",
        default="../data/val_data_siamese/pairsDevTest.txt",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        help="Location where the model should be saved",
        default="../logs",
    )

    args = parser.parse_args()
    train_model(args.train_pairs_path, args.val_pairs_path, args.save_model_dir)
