from output_monkeyrunner_python import OutputMonkeyrunnerPython
from output_arduino import OutputArduino

output = OutputMonkeyrunnerPython()
# output = OutputArduino()

for i in range(0, 2):
    output.tap(200, 100)

