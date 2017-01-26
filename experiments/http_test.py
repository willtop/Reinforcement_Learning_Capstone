import time
from pickle import dumps, loads

import tensorflow as tf
from klein import Klein

from experiments.cnn import cnn

new_height = 96
new_width = 64

x, y, weights, biases, pred = cnn(new_height, new_width)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


class ItemStore(object):
    app = Klein()

    def __init__(self):
        self._items = {}

    @app.route('/', methods=['GET'])
    def items(self, request):
        return 'Hello world'

    @app.route('/', methods=['POST'])
    def items(self, request):
        print('Reading request')
        start_time = time.time()
        request.setHeader('Content-Type', 'application/json')
        body = loads(request.content.read())
        elapsed_time = time.time() - start_time
        print(elapsed_time)

        print('CNN')
        start_time = time.time()
        frame_pred = sess.run(pred, feed_dict={x: [body]})
        elapsed_time = time.time() - start_time
        print(elapsed_time)

        print('Making response')
        start_time = time.time()
        ret = dumps(frame_pred)
        elapsed_time = time.time() - start_time
        print(elapsed_time)

        return ret

if __name__ == '__main__':
    store = ItemStore()
    store.app.run('localhost', 8080)