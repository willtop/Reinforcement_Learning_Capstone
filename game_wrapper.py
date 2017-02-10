import cv2
import numpy as np
from PIL import Image, ImageGrab
from output_processor import output_processor


class Game:
    RENDER_DISPLAY = True
    emulator_resolution = (480, 800)
    bounding_box = (0, 70, 292, 560)  # Daniel Laptop
    # bounding_box = (0, 0, 350, 600)  # Someone else's machine
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]

    def __init__(self, screenshot_dims, game_params, auto_restart=True):
        self.screenshot_dims = screenshot_dims
        self.params = game_params
        self.auto_restart = auto_restart
        self.output_processor = output_processor

    def frame_step(self, input_vec):
        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        raw_im = ImageGrab.grab(bbox=self.bounding_box)
        im = raw_im.resize(self.screenshot_dims, Image.ANTIALIAS)
        screenshot = np.asarray(im)

        chosen_action = np.argmax(input_vec)
        actions = {
            0: lambda: True,
            1: lambda: output_processor.tap(self.__denormalize_screen_position(self.params.tap_position))
        }
        actions[chosen_action]()

        terminal = self.__check_terminal_state(screenshot)
        if self.auto_restart and terminal:
            self.__restart()

        if self.RENDER_DISPLAY:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.screenshot_dims[0] * 3, self.screenshot_dims[1] * 3)
            cv2.imshow('screen', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        return screenshot, terminal

    def __restart(self):
        output_processor.tap(self.__denormalize_screen_position(self.params.restart_tap_position))

    def __check_terminal_state(self, screenshot):
        return all(x in self.get_pixel_color(screenshot, self.params.terminal_pixel_position) for x in self.params.terminal_pixel_color)

    def __denormalize_screen_position(self, position):
        x = int(position[0] * self.emulator_resolution[0])
        y = int(position[1] * self.emulator_resolution[1])
        return x, y

    @staticmethod
    def denormalize_screenshot_position(screenshot, position):
        # numpy arrays are row, colum: so shape is (y, x, channel)
        x = int(position[0] * screenshot.shape[1])
        y = int(position[1] * screenshot.shape[0])
        return x, y

    @staticmethod
    def get_pixel_color(screenshot, position):
        x, y = Game.denormalize_screenshot_position(screenshot, position)
        return screenshot[y, x] # numpy arrays are row, colum: so shape is (y, x, channel)
