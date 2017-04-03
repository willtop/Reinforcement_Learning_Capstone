#!python3

import cv2
import numpy as np
from game_wrapper import Game
from game_params import *


def main():
    game = Game((60, 100), Jason_stack_params_phone, auto_restart=False)

    i = 0

    screenshot,_, terminal = game.frame_step([1, 0])
    cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('screen', screenshot.shape[0] * 3, screenshot.shape[1] * 3)
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('color', 200, 200)

    while True:
        i += 1
        i %= 2

        screenshot, _, terminal = game.frame_step([1, 0])
        pixel_position = game.params.terminal_pixel_position
        for pixel in pixel_position:
          pixel_color = np.copy(Game.get_pixel_color(screenshot, pixel))
          denormalized_pixel_position = Game.denormalize_screenshot_position(screenshot, pixel)
          print(str(denormalized_pixel_position) + ': ' + str(pixel_color))
          screenshot.flags.writeable = True
          screenshot[denormalized_pixel_position] = i * 255

        print(terminal)
        cv2.imshow('screen', cv2.cvtColor(np.transpose(screenshot), cv2.COLOR_GRAY2BGR))
        cv2.imshow('color', cv2.cvtColor(np.asarray([[pixel_color]]), cv2.COLOR_GRAY2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
