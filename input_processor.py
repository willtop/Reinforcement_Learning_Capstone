import cv2
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageFilter
import numpy as np
import time


class GameState:

    bounding_box = (0, 107, 280, 540)
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]

    def __init__(self, dims):
        self.dims = dims
        self.score_box = (100, 25, 190, 85)

    def start(self):
        return

    def restart(self):
        return

    def frame_step(self, input_vec):

        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        start_time = time.time()

        reward = 0
        terminal = False

        raw_im = ImageGrab.grab(bbox=self.bounding_box)

        im = raw_im.resize(self.dims, Image.ANTIALIAS)
        screen = np.asarray(im)

        score_im = np.asarray(raw_im.crop(self.score_box))
        score_im = cv2.cvtColor(score_im, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        score_im = cv2.dilate(score_im, kernel, iterations=1)
        _, score_im = cv2.threshold(score_im, 225, 255, cv2.THRESH_BINARY)

        mask = np.ones(score_im.shape[:2], dtype="uint8") * 255
        _, contours, hierarchy = cv2.findContours(np.copy(score_im), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            #print('area: ' + str(cv2.contourArea(c)))
            if cv2.contourArea(c) < 114:
                cv2.drawContours(mask, [c], -1, 0, -1)
        score_im = cv2.bitwise_and(score_im, score_im, mask=mask)

        score = pytesseract.image_to_string(Image.fromarray(score_im), config='-psm 8 digits_only')
        print('score: ' + str(score))

        elapsed_time = time.time() - start_time
        print(elapsed_time)

        if True:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.dims[0]*3, self.dims[1]*3)
            cv2.imshow('screen',  cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('score', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('score', (self.score_box[2]-self.score_box[0])*3, (self.score_box[3]-self.score_box[1])*3)
            cv2.imshow('score', cv2.cvtColor(score_im, cv2.COLOR_GRAY2BGR))

            # cv2.imshow('mask', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        return screen, reward, terminal


if __name__ == '__main__':
    gameState = GameState((80, 120))

    while True:
        gameState.frame_step([0, 1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
