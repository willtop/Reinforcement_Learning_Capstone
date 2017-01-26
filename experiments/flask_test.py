import json

import tensorflow as tf
from flask import Flask
from flask import make_response

from experiments.cnn import cnn

new_height = 96
new_width = 64

app = Flask(__name__)

x, y, weights, biases, pred = cnn(new_height, new_width)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/cnn', methods=['POST'])
def echo():
    # if not request.json:
    #     abort(400)

    # print(str(request.data)[1:])
    # frame = json.loads(str(request.data)[1:])
    # print(frame)
    #
    # frame_pred = sess.run(pred, feed_dict={x: [frame]})

    return make_response(json.dumps("{'test': 'hello'}"), 200)


if __name__ == "__main__":
    app.run(threaded=True)