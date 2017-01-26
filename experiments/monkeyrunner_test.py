import time
from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
 
# Connects to the first device available through the adb tool
print('Connecting...')
device = MonkeyRunner.waitForConnection()
 
# print 'Touch...'
# start_time = time.time()
# device.touch(730, 2245, MonkeyDevice.DOWN_AND_UP)
# elapsed_time = time.time() - start_time
# print elapsed_time

print('Taking screenshot...')

# take a screenshot
start_time = time.time()
screenshot = device.takeSnapshot()
elapsed_time = time.time() - start_time
print(elapsed_time)

print('Converting...')
start_time = time.time()
bytes = screenshot.convertToBytes()
elapsed_time = time.time() - start_time
print(elapsed_time)

# print bytes
