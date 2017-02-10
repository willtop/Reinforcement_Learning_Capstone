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

    def __init__(self, screenshot_dims, params, auto_restart=True):
        self.screenshot_dims = screenshot_dims
        self.params = params
        self.auto_restart = auto_restart
        self.output_processor = output_processor

    def frame_step(self, input_vec):
        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        raw_im = ImageGrab.grab(bbox=self.bounding_box)
        im = raw_im.resize(self.screenshot_dims, Image.ANTIALIAS)
        im = im.convert(mode="L")
        screenshot = np.asarray(im)
        screenshot = np.transpose(screenshot)

        chosen_action = np.argmax(input_vec)
        actions = {
          0: lambda: True,
          1: lambda: output_processor.tap((self.__denormalize_screen_position(self.params.tap_position)))
        }
        actions[chosen_action]()
        
        # print("ACTION: {}".format(chosen_action))

        terminal = self.__check_terminal_state(screenshot)
        if self.auto_restart and terminal:
            self.restart()

        if self.RENDER_DISPLAY:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.screenshot_dims[0] * 3, self.screenshot_dims[1] * 3)
            cv2.imshow('screen', cv2.cvtColor(np.transpose(screenshot), cv2.COLOR_GRAY2BGR))
            cv2.waitKey(1)

        return screenshot, terminal

    def restart(self):
        print('>>>>>>> RESTART')
        output_processor.tap(self.__denormalize_screen_position(self.params.restart_tap_position))

    def __check_terminal_state(self, screenshot):
        return self.get_pixel_color(screenshot, self.params.terminal_pixel_position) == self.params.terminal_pixel_color

    def __denormalize_screen_position(self, position):
        x = int(position[0] * self.emulator_resolution[0])
        y = int(position[1] * self.emulator_resolution[1])
        return x, y

    @staticmethod
    def denormalize_screenshot_position(screenshot, position):
        x = int(position[0] * screenshot.shape[0])
        y = int(position[1] * screenshot.shape[1])
        return x, y

    @staticmethod
    def get_pixel_color(screenshot, position):
        x, y = Game.denormalize_screenshot_position(screenshot, position)
        return screenshot[x, y]
