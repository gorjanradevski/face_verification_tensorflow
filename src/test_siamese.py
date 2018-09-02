import cv2
import tensorflow as tf
import os
import numpy as np
import argparse
from time import sleep

from utils.global_config import IMG_HEIGHT, IMG_WIDTH
from siamese_net import SiameseNet


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def test_predictions(args):
    input_image1 = np.expand_dims(
        cv2.resize(cv2.imread(args.input_image1), (IMG_HEIGHT, IMG_WIDTH)), axis=0
    )
    input_image2 = np.expand_dims(
        cv2.resize(cv2.imread(args.input_image2), (IMG_HEIGHT, IMG_WIDTH)), axis=0
    )
    model = SiameseNet()
    loader = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        sess.run(tf.global_variables_initializer())
        model.restore_from_checkpoint(sess, loader, args.checkpoint_path)

        feed_dict_inference = {
            model.input_images1: input_image1,
            model.input_images2: input_image2,
            model.is_training: False,
        }
        predictions = sess.run(model.prediction, feed_dict_inference)
        print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script prints the similarity of the two images"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Location the model checkpoint is",
        default="../logs/siamese_net",
    )
    parser.add_argument(
        "--input_image1",
        type=str,
        help="Location where the first image is",
    )
    parser.add_argument(
        "--input_image2",
        type=str,
        help="Location where the second image is",
    )

    args = parser.parse_args()
    test_predictions(args)
