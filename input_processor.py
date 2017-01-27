import cv2
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageFilter
import numpy as np

# pytesseract.pytesseract.tesseract_cmd = 'pytesser_v0.0.1/tesseract.exe'

class GameState:

    bounding_box = (0, 107, 562, 1080)
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]

    def __init__(self, dims):
        self.dims = dims
        self.score_box = (200, 100, 400, 200)

    def start(self):
        return

    def restart(self):
        return

    def frame_step(self, input_vec):

        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        reward = 0
        terminal = False

        raw_im = ImageGrab.grab(bbox=self.bounding_box)

        im = raw_im.resize(self.dims, Image.ANTIALIAS)
        screen = np.asarray(im)

        score_im = raw_im.crop(self.score_box)
        score_im = score_im.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(score_im)
        score_im = enhancer.enhance(10)
        score = pytesseract.image_to_string(score_im, config='digits_only')
        print(score)

        if True:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.dims[0]*3, self.dims[1]*3)
            cv2.imshow('screen',  cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('score', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('score', (self.score_box[2]-self.score_box[0])*3, (self.score_box[3]-self.score_box[1])*3)
            cv2.imshow("score", cv2.cvtColor(np.asarray(score_im), cv2.COLOR_RGB2BGR))

        return screen, reward, terminal


if __name__ == '__main__':
    gameState = GameState((80, 120))

    while True:
        gameState.frame_step([0, 1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
