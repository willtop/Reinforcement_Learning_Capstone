import cv2
import time
import numpy as np
import android_game_params
from PIL import Image, ImageGrab
from android_output_processor import output_processor

selected_params = android_game_params.flappy_cat_params

NAME = selected_params.name
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000

EXPLORE = 100000
FINAL_EPSILON = 0.0
INITIAL_EPSILON = 1.0
REPLAY_MEMORY = 10000
REPLAY_MEMORY_DISCARD_AMOUNT = 100
BATCH = 100


class GameState:
    RENDER_DISPLAY = False
    emulator_resolution = (240, 400)
    bounding_box = (1, 71, 121, 271)  # Daniel Laptop
    # bounding_box = (20, 100, 380, 700)  # Jason's laptop - Emulator

    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]
    
    TERMINAL_PIXEL_TOLERANCE = 3

    def __init__(self):
        self.screenshot_dims = (80, 80)
        self.params = selected_params
        self.output_processor = output_processor

    def frame_step(self, input_vec):
        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        raw_im = ImageGrab.grab(bbox=self.bounding_box)
        im = raw_im.resize(self.screenshot_dims, Image.ANTIALIAS)
        screenshot = np.asarray(im)

        reward = 1
        terminal = self.__check_terminal_state(screenshot)
        if terminal:
            reward = -100
            self.__restart()
        else:
            chosen_action = np.argmax(input_vec)
            actions = {
              0: lambda: True,
              1: lambda: output_processor.tap((self.__denormalize_screen_position(self.params.tap_position)))
            }
            actions[chosen_action]()

        if self.RENDER_DISPLAY:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.screenshot_dims[0] * 3, self.screenshot_dims[1] * 3)
            cv2.imshow('screen', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return screenshot, reward, terminal

    def __restart(self):
        output_processor.tap(self.__denormalize_screen_position(self.params.restart_tap_position))
        time.sleep(0.5)

    def __check_terminal_state(self, screenshot):
        check = True
        for pixel, color in zip(self.params.terminal_pixel_position, self.params.terminal_pixel_color):
            pixel_color = self.get_pixel_color(screenshot, pixel)
            for c in range(0, 3):
                if pixel_color[c] < color[c] - self.TERMINAL_PIXEL_TOLERANCE or \
                        pixel_color[c] > color[c] + self.TERMINAL_PIXEL_TOLERANCE:
                    check = False
        return check

    def __denormalize_screen_position(self, position):
        x = int(position[0] * self.emulator_resolution[0])
        y = int(position[1] * self.emulator_resolution[1])
        return x, y

    @staticmethod
    def denormalize_screenshot_position(screenshot, position):
        # Python image conventions are y, x
        x = int(position[1] * screenshot.shape[1])
        y = int(position[0] * screenshot.shape[0])
        return x, y

    @staticmethod
    def get_pixel_color(screenshot, position):
        x, y = GameState.denormalize_screenshot_position(screenshot, position)
        return screenshot[x, y]
