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
    haar_face_cascade = cv2.CascadeClassifier(args.haar_classifier_path)
    model = ConvNet()
    loader = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        sess.run(tf.global_variables_initializer())
        model.restore_from_checkpoint(sess, loader, args.checkpoint_path)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            faces = haar_face_cascade.detectMultiScale(
                frame, scaleFactor=1.1, minNeighbors=5
            )
            if len(faces) == 1 and ret == True:
                (x, y, w, h) = faces[0]
                croped_img = frame[y : y + h, x : x + w]
                if croped_img.shape[0] > 0 and croped_img.shape[1] > 0:
                    input_image = np.expand_dims(
                        cv2.resize(croped_img, (IMG_HEIGHT, IMG_WIDTH)), axis=0
                    )
                    feed_dict_inference = {
                        model.input_images: input_image,
                        model.is_training: False,
                    }
                    predictions = sess.run(model.prediction, feed_dict_inference)
                    print(predictions)
                    cv2.imshow("Checking images", croped_img)

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
    parser.add_argument(
        "--haar_classifier_path",
        type=str,
        help="Location the model checkpoint is",
        default="../haar_classifiers/haarcascade_frontalface_alt.xml",
    )

    args = parser.parse_args()
    perform_inference_on_camera_input(args)
