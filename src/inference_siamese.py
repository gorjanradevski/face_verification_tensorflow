import cv2
import tensorflow as tf
import os
import numpy as np
import argparse
from time import sleep

from utils.global_config import IMG_HEIGHT, IMG_WIDTH
from siamese_net import SiameseNet


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def perform_inference_on_camera_input(args):
    # 85 -> 235 WIDTH
    # 45 -> 195 HEIGHT
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 150)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)
    anchor_image = np.expand_dims(cv2.imread(args.anchor_image_path), axis=0)
    model = SiameseNet()
    loader = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        sess.run(tf.global_variables_initializer())
        model.restore_from_checkpoint(sess, loader, args.checkpoint_path)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            cropped_frame = frame[45:195,85:235]
            input_image = np.expand_dims(cropped_frame, axis=0)

            feed_dict_inference = {
                model.input_images1: input_image,
                model.input_images2: anchor_image,
                model.is_training: False,
            }
            predictions = sess.run(model.prediction, feed_dict_inference)
            print(predictions)
            cv2.imshow("Checking images", cropped_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sleep(1)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script takes a path to the" "checkpoint and the anchor image"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Location the model checkpoint is",
        default="../logs/siamese_net",
    )
    parser.add_argument(
        "--anchor_image_path",
        type=str,
        help="Location where the validation pairs are",
        default="../data/anchor_image.jpg",
    )

    args = parser.parse_args()
    perform_inference_on_camera_input(args)
