import cv2
import numpy as np
from game_wrapper import Game
from game_params import *


def main():
    game = Game((100, 166), stack_params)

    i = 0

    screenshot, terminal = game.frame_step([0, 1])
    cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('screen', screenshot.shape[1] * 3, screenshot.shape[0] * 3)
    cv2.namedWindow('color', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('color', 200, 200)

    while True:
        i += 1
        i %= 2

        screenshot, terminal = game.frame_step([0, 1])
        pixel_position = game.params.terminal_pixel_position
        pixel_color = np.copy(Game.get_pixel_color(screenshot, pixel_position))
        denormalized_pixel_position = Game.denormalize_screenshot_position(screenshot, pixel_position)
        print(str(denormalized_pixel_position) + ': ' + str(pixel_color))
        print(terminal)
        screenshot.flags.writeable = True
        screenshot[denormalized_pixel_position[::-1]] = np.multiply([i, i, i], 255)

        cv2.imshow('screen', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        cv2.imshow('color', cv2.cvtColor(np.asarray([[pixel_color]]), cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
