import re
import sys
import time
import os

from PIL import Image

start_time = time.time()

from com.dtmilano.android.viewclient import ViewClient

device, serialno = ViewClient.connectToDeviceOrExit()

elapsed_time = time.time() - start_time
print(elapsed_time)

# print("################## Settings application test ####################")
# device.startActivity('com.android.settings/.Settings')
# print('SUCCESFULLY OPENED SETTINGS APP')
# vc.dump() ## this is used to refresh the screen.


for i in range(0, 10):
    start_time = time.time()

    device.touch(240, 100)

    elapsed_time = time.time() - start_time
    print(elapsed_time)