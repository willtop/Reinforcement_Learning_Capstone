import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageGrab

from experiments.cnn import cnn

bounding_box = (0, 107, 562, 1080)
width = bounding_box[2] - bounding_box[0]
height = bounding_box[3] - bounding_box[1]
new_height = 96
new_width = 64
# new_width = int(new_height * width / height)


def main():

    x, y, weights, biases, pred = cnn(new_height, new_width)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        while True:
            print('Screencap + resize')
            start_time = time.time()
            im = ImageGrab.grab(bbox=bounding_box)
            im = im.resize((new_width, new_height), Image.ANTIALIAS)
            frame = np.asarray(im)
            elapsed_time = time.time() - start_time
            print(elapsed_time)

            print('Local CPU CNN')
            start_time = time.time()
            frame_pred = sess.run(pred, feed_dict={x: [frame]})
            elapsed_time = time.time() - start_time
            print(elapsed_time)

            # print('Remote GPU CNN')
            # start_time = time.time()
            # r = requests.post("http://127.0.0.1:8080", dumps(frame.tolist()))
            # frame_pred = loads(r.content)
            # print(frame_pred)
            # elapsed_time = time.time() - start_time
            # print(elapsed_time)

            # Display the resulting frame
            cv2.imshow('Video', frame[..., [2, 1, 0]]) # Convert to BGR for OpenCV

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()