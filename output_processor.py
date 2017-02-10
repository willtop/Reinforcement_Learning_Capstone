#!python3
from output_monkeyrunner_python import OutputMonkeyrunnerPython
from output_arduino import OutputArduino
import pyautogui

# output_processor = OutputMonkeyrunnerPython()
# output_processor = OutputArduino()
def output_processor(pos):
  pyautogui.click(pos[0],pos[1])

if __name__ == '__main__':
    for i in range(0, 2):
        # output_processor.tap(200, 100)
        output_processor((240, 400))

