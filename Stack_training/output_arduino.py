#! python3

import serial
import struct
import time

class OutputArduino:
    def __init__(self):
        #Start the serial port to communicate with arduino
        self.data = serial.Serial('com3',9600, timeout=1)
        time.sleep(2)
        self.data.write(struct.pack('>B',55))

    def tap(self, point):
        self.data.write(struct.pack('>B',35))
        time.sleep(0.1)
        self.data.write(struct.pack('>B',55))
        
        
if __name__ == '__main__':
    Arduino = OutputArduino()
    print('Sent tap at:')
    print(time.time())
    Arduino.tap((540, 960))
    # while 1:
        # pass
