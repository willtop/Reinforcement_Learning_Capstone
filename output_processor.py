from output_monkeyrunner_python import OutputMonkeyrunnerPython
from output_arduino import OutputArduino

output_processor = OutputMonkeyrunnerPython()
# output_processor = OutputArduino()

if __name__ == '__main__':
    for i in range(0, 2):
        output_processor.tap(200, 100)

