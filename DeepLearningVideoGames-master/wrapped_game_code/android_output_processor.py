#!python3
from android_output_monkeyrunner_client import OutputMonkeyrunner

output_processor = OutputMonkeyrunner()

if __name__ == '__main__':
    for i in range(0, 2):
        output_processor.tap(200, 100)

