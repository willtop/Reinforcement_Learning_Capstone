"""
This file must be run with the monkeyrunner tool from the Android SDK.
"""

import time
import socket
from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice

DATA_END = '|'

print('Connecting to MonkeyRunner...')
device = MonkeyRunner.waitForConnection()

print('Starting socket server...')
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 1234))
server_socket.listen(5)

print('Waiting for client to connect...')
(client_socket, address) = server_socket.accept()
print('Client connected.')

buffer = ''
while 1:
    buffer += client_socket.recv(1024).decode()
    split_index = buffer.find(DATA_END)
    if len(buffer) > 0 and split_index != -1:
        # Read just one message
        split_buffer = buffer.split(DATA_END)
        data = split_buffer[0]
        buffer = buffer[split_index+1:]
        # Process message
        print('Received at:')
        print(time.time())
        print(data)
        parts = data.split(',')
        x = int(parts[0])
        y = int(parts[1])

        device.touch(x, y, MonkeyDevice.DOWN_AND_UP)
