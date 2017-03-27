import cv2
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageFilter
import numpy as np
import time
# Debug importing
from game_wrapper import Game
from game_params import *

# define a global variable record the current score 
# current_score = 0


class ScoreCalc:
  def __init__(self):
    self.current_score = 0
    # Debug constants (used for debugging)
    self.score_box = [130, 80, 200, 167]
    
  def resertScore(self):
    self.current_score = 0
    
  def getScore(self, score_im):
      
    kernel = np.ones((3, 3), np.uint8)
    score_im = cv2.dilate(score_im, kernel, iterations=1)
    _, score_im = cv2.threshold(score_im, 225, 255, cv2.THRESH_BINARY)
    # print(type(score_im))
    # expand the score_im horizontally
    score_width_ori, score_height_ori = np.shape(score_im)
    #score_im = np.resize(score_im, (score_width_ori*2, score_height_ori*2))
    #print(np.shape(score_im))
     
    mask = np.ones(score_im.shape[:2], dtype="uint8") * 255
    _, contours, hierarchy = cv2.findContours(np.copy(score_im), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        #print('area: ' + str(cv2.contourArea(c)))
        if cv2.contourArea(c) < 114:
            cv2.drawContours(mask, [c], -1, 0, -1)
    score_im = cv2.bitwise_and(score_im, score_im, mask=mask)

    cv2.namedWindow('score_debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('score_debug', (self.score_box[2]-self.score_box[0])*3, (self.score_box[3]-self.score_box[1])*3)
    cv2.imshow('score_debug', cv2.cvtColor(score_im, cv2.COLOR_GRAY2BGR))
    cv2.waitKey(1)
    
    try:
      score = pytesseract.image_to_string(Image.fromarray(score_im).resize((score_width_ori*2, score_height_ori*2), Image.ANTIALIAS), config='-psm 8 digits_only')
      # print("Raw Score: {}".format(score))
    except UnicodeDecodeError:
      return -1
 
    # The score is always confirmed to be with str type
    # Try to convert into integer type, upon exception don't do anything
    polished_score = []
    final_score = []

    # As 1 is always confused with I, convert it as pre-processing
    for i in range(len(score)):
        if(score[i] == 'I' or score[i] == 'i' or score[i] == 'l' or score[i] == '{' or \
           score[i] == '}' or score[i] == 'L' or score[i] == '|' or score[i] == '!'):
        # detect a 1 here
            polished_score.append(1)
        # elif (score[i] == 'A'):
            # polished_score.append(4)
        elif (score[i] == 'O' or score[i] == 'o'):
            polished_score.append(0)
        else:
            polished_score.append(score[i])
            
    if score == '':
        polished_score.append(0)

    for raw_digit in polished_score:
        try:
            digit = int(raw_digit)
            final_score.append(digit)
            # # always check if it reads one point increase
            # if(int(''.join(map(str, final_score))) == self.current_score+1):
            #     # we have reached a promising reading
            #     break					
        except ValueError:
            return -1

    valid = False
    # if obtaining a valid score:
    if(len(final_score) > 0):
        if(int(''.join(map(str, final_score))) >= self.current_score ):
            self.current_score = int(''.join(map(str, final_score)))
            valid = True
    #print("raw: {}".format(int(''.join(map(str, final_score)))))
        # additional fix for starting with 1
        #if(int(''.join(map(str, final_score))) + 10 >= self.current_score):
        #   self.current_score = int(''.join(map(str, final_score))) + 10
        #    valid = True


    if valid:
        print("At input processor: Processed score: {}".format(self.current_score))
        return self.current_score
    return -1


# self debug function to mainly run OCR
if __name__ == '__main__':
    # call game wrapper to obtain score image
    game = Game((60, 100), boom_dots_params, auto_restart=False)
    scoreCalc = ScoreCalc()
    for i in range(30):
        _, score_im, terminal = game.frame_step([0, 1])
        score = scoreCalc.getScore(score_im)
        print("state score: {}, terminal? {}".format(score, terminal))