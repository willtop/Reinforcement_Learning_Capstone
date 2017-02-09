import cv2

import numpy as np
from PIL import Image, ImageGrab
from output_processor import output_processor


class Game:
    RENDER_DISPLAY = True
    bounding_box = (0, 0, 350, 600)
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]

    '''
    terminal_state_detector = {
        [x, y]: [r, g, b]
    }
    '''

    def __init__(self, dims, tap_position, restart_tap_position, terminal_state_detector):
        self.dims = dims
        self.tap_position = tap_position
        self.restart_tap_position = restart_tap_position
        self.terminal_state_detector = terminal_state_detector
        self.output_processor = output_processor

    def frame_step(self, input_vec):
        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        raw_im = ImageGrab.grab(bbox=self.bounding_box)
        im = raw_im.resize(self.dims, Image.ANTIALIAS)
        screenshot = np.asarray(im)

        chosen_action = np.argmax(input_vec)
        actions = {
            0: lambda: None,
            1: output_processor.tap(self.tap_position[0], self.tap_position[1])
        }
        actions[chosen_action]()

        terminal = self.__check_terminal_state(screenshot)
        if terminal:
            self.__restart()

        if self.RENDER_DISPLAY:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.dims[0] * 3, self.dims[1] * 3)
            cv2.imshow('screen', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        return screenshot, terminal

    def __restart(self):
        output_processor.tap(self.restart_tap_position[0], self.restart_tap_position[1])

    def __check_terminal_state(self, screenshot):
        for pixel in self.terminal_state_detector:
            if screenshot[pixel] != self.terminal_state_detector[pixel]:
                return False
        return True
