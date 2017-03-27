#!python3
from output_monkeyrunner_python import OutputMonkeyrunnerPython
from output_arduino import OutputArduino
from output_pyautogui import OutputPyautogui

# output_processor = OutputMonkeyrunnerPython()
# output_processor = OutputArduino()
output_processor = OutputPyautogui()

# The module wouldn't be invoked
# if __name__ == '__main__':
#     for i in range(0, 2):
#         output_processor.tap(200, 100)

