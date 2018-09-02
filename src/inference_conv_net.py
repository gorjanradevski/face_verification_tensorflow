import cv2
import tensorflow as tf
import os
import numpy as np
import argparse
from time import sleep

from utils.global_config import IMG_HEIGHT, IMG_WIDTH
from conv_net import ConvNet


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def perform_inference_on_camera_input(args):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)
    model = ConvNet()
    loader = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        sess.run(tf.global_variables_initializer())
        model.restore_from_checkpoint(sess, loader, args.checkpoint_path)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            input_image = np.expand_dims(
                cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH)), axis=0
            )

            feed_dict_inference = {
                model.input_images: input_image,
                model.is_training: False,
            }
            predictions = sess.run(model.prediction, feed_dict_inference)
            print(predictions)
            cv2.imshow("Checking images", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sleep(1)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Only the path to the checkpoint is required"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Location the model checkpoint is",
        default="../logs/conv_net",
    )

    args = parser.parse_args()
    perform_inference_on_camera_input(args)
