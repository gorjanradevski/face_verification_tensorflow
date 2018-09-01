import cv2
import tensorflow as tf
import os
import numpy as np
import argparse

from utils.global_config import IMG_HEIGHT, IMG_WIDTH
from siamese_net import SiameseNet


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def perform_inference_on_camera_input(args):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)
    anchor_image = cv2.imread(args.anchor_image_path)
    model = SiameseNet()
    with tf.Session() as sess:
        # Restore variables from disk.
        model.restore_from_checkpoint(sess, args.checkpoint_path)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            check_image = np.array(cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH)))
            feed_dict_inference = {
                model.input_images1: check_image,
                model.input_images2: anchor_image,
                model.is_training: False,
            }
            predictions = sess.run(model.inference, feed_dict_inference)
            print(predictions)
            cv2.imshow("Face image", check_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script takes a path where the"
        "the checkpoint is and the anchor image"
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
