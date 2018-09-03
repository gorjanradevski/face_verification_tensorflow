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


def train_model(pairs_dataset: str, save_model_path: str):
    """

    Args:
        data_dir: Where the dataset is

    Returns:

    """
    dataset = load_all_image_paths_siamese(pairs_dataset)
    # Artificially modifying the dataset for better results
    train_image_paths = dataset[:5400]
    val_image_paths = dataset[5400:]

    log.info(f"{len(train_image_paths)} images belonging to the train set...")
    log.info(f"{len(val_image_paths)} images belonging to the validation set...")
    model = SiameseNet()
    log.info("Model built...")
    BEST_VAL_LOSS = sys.maxsize

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPOCHS):
            train_loss_epoch = 0.0
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

                train_loss_epoch += train_loss

                full_epoch_train.set_description(
                    f"Loss for epoch {e+1}: %g" % train_loss_epoch
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
                print("===============================================")
                print(f"Found new best! Saving model on epoch {e+1}...")
                print("===============================================")
                saver.save(sess, f"{save_model_path}")
                BEST_VAL_LOSS = validation_loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script takes a path where the"
        "train pairs are as well as where the validation pairs are"
    )
    parser.add_argument(
        "--pairs_dataset",
        type=str,
        help="Location the training-validations are",
        default="../data/pairs.txt",
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        help="Location where the model should be saved",
        default="../logs/siamese_net2",
    )

    args = parser.parse_args()
    train_model(args.pairs_dataset, args.save_model_path)
