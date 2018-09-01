import cv2
import tensorflow as tf
import os
import numpy as np

from utils.global_config import IMG_HEIGHT, IMG_WIDTH


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cap = cv2.VideoCapture(0)
count = 0

with tf.Session() as sess:
    # Restore variables from disk.
    loader = tf.train.import_meta_graph('logs/siamese_net.meta')
    loader.restore(sess, 'logs/siamese_net')

    true = tf.get_default_graph().get_tensor_by_name("inputs1:0")
    test = tf.get_default_graph().get_tensor_by_name("inputs2:0")
    softmax = tf.get_default_graph().get_tensor_by_name("Softmax:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("Placeholders/dropout_prob:0")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if count % 30 == 0:
            resize_img = np.array(cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH)))
            predictions = sess.run(softmax, {X: image, keep_prob: 1.0})
            cv2.imshow("Face image", resize_img)
            print(predictions)
            if predictions[0][0] == 1:
                print('Gorjan is in the picture!!!!')
            else:
                print('You are not Gorjan...')

            count = 0
            else:
                print('Zero or more faces in a picture')
                count = 0

        count+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()