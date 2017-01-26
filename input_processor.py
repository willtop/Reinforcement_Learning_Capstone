import cv2
import numpy as np


class GameState:
    def __init__(self):
        self.screen = np.ones((100, 100), np.float32) * 255
        self.screen = cv2.cvtColor(self.screen, cv2.COLOR_GRAY2BGR)
        self.steps = 0

    def frame_step(self, input_vec):

        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        reward = -1

        if self.screen[0,0,0] == 255:
            if input_vec[1] == 1:
                reward = 1
                self.screen *= (1./255.)
            elif input_vec[0] == 1:
                reward = 0
        else:
            if input_vec[2] == 1:
                reward = 1
                self.screen *= 255
            elif input_vec[0] == 1:
                reward = 0

        self.steps += 1
        terminal = False
        if self.steps >= 10:
            self.steps = 0
            terminal = True

        return self.screen, reward, terminal