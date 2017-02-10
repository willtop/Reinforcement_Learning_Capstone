# First of all, introduce pysseract into the python path
# import sys
# sys.path.append('C:\\Users\\happy\\AppData\\Local\\Tesseract-OCR')
# print("Done setting pysseract script into path!")

import cv2
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageFilter
import numpy as np

# pytesseract.pytesseract.tesseract_cmd = 'pytesser_v0.0.1/tesseract.exe'

# define a global variable record the current score 
current_score = 0


class GameState:
    # bounding_box: original setting 0, 107, 562, 1080
    bounding_box = (0, 0, 350, 600)
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]

    def __init__(self, dims):
        self.dims = dims
        # score_box: original setting 200, 100, 400, 200
        self.score_box = (140, 120, 200, 195)

    def start(self):
        return

    def restart(self):
        return

    def frame_step(self, input_vec):

        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        # reference to a global variable
        global current_score

        reward = 0
        terminal = False

        raw_im = ImageGrab.grab(bbox=self.bounding_box)

        im = raw_im.resize(self.dims, Image.ANTIALIAS)
        screen = np.asarray(im)

        score_im = raw_im.crop(self.score_box)
        score_im = score_im.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(score_im)
        score_im = enhancer.enhance(20)
        score_im = Image.fromarray(cv2.cvtColor(np.asarray(score_im), cv2.COLOR_RGB2GRAY))
        score = pytesseract.image_to_string(score_im, config='-psm 7 digits_only')
        print(score)
        # print("try convert score")
        # print(type(score))

        # The score is always confirmed to be with str type
        # Try to convert into integer type, upon exception don't do anything
        polished_score = []
        final_score = []

        # As 1 is always confused with I, convert it as pre-processing
        for i in range(len(score)):
            if (score[i] == 'I'):
                # detect a 1 here
                polished_score.append(1)
            else:
                polished_score.append(score[i])

        for raw_digit in polished_score:
            try:
                digit = int(raw_digit)
                final_score.append(digit)
                # always check if it reads one point increase
                if (int(''.join(map(str, final_score))) == current_score + 1):
                    # we have reached a promising reading
                    break
            except ValueError:
                pass

        # if obtaining a valid score:
        if (len(final_score) > 0):
            if (int(''.join(map(str, final_score))) > current_score and int(
                    ''.join(map(str, final_score))) <= current_score + 3):
                current_score = int(''.join(map(str, final_score)))

        print(current_score)

        # render the displays
        if True:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.dims[0] * 3, self.dims[1] * 3)
            cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('score', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('score', (self.score_box[2] - self.score_box[0]) * 3,
                             (self.score_box[3] - self.score_box[1]) * 3)
            cv2.imshow("score", cv2.cvtColor(np.asarray(score_im), cv2.COLOR_GRAY2BGR))

        return screen, reward, terminal


if __name__ == '__main__':
    gameState = GameState((120, 180))

    while True:
        gameState.frame_step([0, 1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
