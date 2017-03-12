import socket
import time


DATA_END = '|'


class OutputMonkeyrunner:

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('localhost', 1234))

    def tap(self, point):
        message = str(point[0]) + ',' + str(point[1]) + DATA_END
        self.socket.send(message.encode())

if __name__ == '__main__':
    monkey = OutputMonkeyrunner()
    print('Sent tap at:')
    print(time.time())
    monkey.tap((200, 100))
    while 1:
        pass