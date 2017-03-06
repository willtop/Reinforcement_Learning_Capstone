import cv2
import numpy as np
from PIL import Image, ImageGrab
from output_processor import output_processor
import win32gui
import time

class Game:
    RENDER_DISPLAY = True
    emulator_resolution = (480, 800)
    # corner = (0,0)
    # bounding_box = (0, 70, 292, 560)  # Daniel Laptop
    # bounding_box = (20, 100, 380, 700)  # Jason's laptop - Emulator
    
    # Jason's Leapdroid Settings
    # corner = win32gui.GetWindowRect(win32gui.FindWindow(None,"Leapdroid"))
    corner = win32gui.GetWindowRect(win32gui.FindWindow(None,"Nox-1"))
    bounding_box = (corner[0], corner[1]+40, corner[0]+emulator_resolution[0], corner[1]+emulator_resolution[1])
    # print("Found box: {}".format(bounding_box))
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]
    
    #Terminal state check pixel colour tolerance
    tolerance = 5

    def __init__(self, screenshot_dims, params, auto_restart=True):
        self.screenshot_dims = screenshot_dims
        self.params = params
        self.auto_restart = auto_restart
        self.output_processor = output_processor

    def frame_step(self, input_vec):
        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        raw_im = ImageGrab.grab(bbox=self.bounding_box)
        score_im = np.asarray(raw_im.crop(self.params.score_box))
        score_im = cv2.cvtColor(score_im, cv2.COLOR_RGB2GRAY)
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
        
        #Check location of terminal position
        # screenshot.flags['WRITEABLE'] = True
        # terminal_pos = [self.denormalize_screenshot_position(screenshot,v) for v in self.params.terminal_pixel_position]
        # for i,pos in enumerate(terminal_pos):
          # print("pos {}: {}".format(i,screenshot[pos[0],pos[1]]))
          # screenshot[pos[0],pos[1]]=0
        
        if self.RENDER_DISPLAY:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.screenshot_dims[0] * 3, self.screenshot_dims[1] * 3)
            cv2.imshow('screen', cv2.cvtColor(np.transpose(screenshot), cv2.COLOR_GRAY2BGR))
            cv2.namedWindow('score', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('score', (self.params.score_box[2]-self.params.score_box[0])*3, (self.params.score_box[3]-self.params.score_box[1])*3)
            cv2.imshow('score', cv2.cvtColor(score_im, cv2.COLOR_GRAY2BGR))
            cv2.waitKey(1)

        return screenshot, score_im, terminal

    def restart(self):
        print('>>>>>>> RESTART')
        time.sleep(0.1)
        output_processor.tap(self.__denormalize_screen_position(self.params.restart_tap_position))
        time.sleep(0.2)
        output_processor.tap(self.__denormalize_screen_position(self.params.restart_tap_position))

    def __check_terminal_state(self, screenshot):
        check = True
        for pixel, colour in zip(self.params.terminal_pixel_position, self.params.terminal_pixel_color):
            if self.get_pixel_color(screenshot, pixel) < colour-self.tolerance or self.get_pixel_color(screenshot, pixel) > colour+self.tolerance:
                check = False
        return  check

    def __denormalize_screen_position(self, position):
        x = int(position[0] * self.emulator_resolution[0] + self.corner[0])
        y = int(position[1] * self.emulator_resolution[1] + self.corner[1])
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
