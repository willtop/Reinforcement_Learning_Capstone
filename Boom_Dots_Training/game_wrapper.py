import cv2
import numpy as np
from PIL import Image, ImageGrab
from output_processor import output_processor
import time

class Game:
    RENDER_DISPLAY = False
    emulator_resolution = (485, 770)    
    bounding_box = (0, 105, 480, 959)  # Daniel Laptop
    # bounding_box = (0, 0, 485, 770) # for getting game field, the emulator screen
    corner = (0,0) # upper left corner placement of the game field
    # print("Found box: {}".format(bounding_box))
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]
    
    #Terminal state check pixel colour tolerance
    tolerance = 5

    def __init__(self, screenshot_dims, params, auto_restart=False):
        self.screenshot_dims = screenshot_dims
        self.params = params
        self.auto_restart = auto_restart
        self.output_processor = output_processor

    def frame_step(self, input_vec):
        if sum(input_vec) != 1:
            raise ValueError('Multiple input actions!')

        # think should perform the action first
        chosen_action = np.argmax(input_vec)
        actions = {
          0: lambda: True,
          1: lambda: output_processor.tap((self.__denormalize_screen_position(self.params.tap_position)))
        }
        actions[chosen_action]()
        #if(chosen_action):
        #    print("just tapped!")

        # shouldn't wait here for a bit?
        # Don't put delay: to ensure high enough sampling rate for optimal state
        #time.sleep(0.25)

        raw_im = ImageGrab.grab(bbox=self.bounding_box)
        # score_im = np.asarray(raw_im.crop(self.params.score_box))
        # score_im = cv2.cvtColor(score_im, cv2.COLOR_RGB2GRAY)
        im = raw_im.resize(self.screenshot_dims, Image.ANTIALIAS) #reshape the game field for CNN input
        im = im.convert(mode="L")
        screenshot = np.asarray(im)
        screenshot = np.transpose(screenshot)

        #print("Action Performed: {}".format(chosen_action))
        #increment = self.__check_scoring(screenshot)

        terminal = self.__check_terminal_state(screenshot)
        # Comment out the if statement to save time
        # if self.auto_restart and terminal:
        #    self.restart()
        #if not self.auto_restart and terminal:
            #print("Game ends. Not restarting.")
        
        # Check location of terminal position
        # screenshot.flags['WRITEABLE'] = True
        # scoring_pos = [self.denormalize_screenshot_position(screenshot,v) for v in self.params.terminal_pixel_position]
        # for i,pos in enumerate(scoring_pos):
        #     print("pos {}: {}".format(i,screenshot[pos[0],pos[1]]))
        #     screenshot[pos[0],pos[1]]=0 # set the position to black so easily visible
        
        if self.RENDER_DISPLAY:
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', self.screenshot_dims[0] * 3, self.screenshot_dims[1] * 3)
            cv2.imshow('screen', cv2.cvtColor(np.transpose(screenshot), cv2.COLOR_GRAY2BGR))
            # cv2.namedWindow('score', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('score', (self.params.score_box[2]-self.params.score_box[0])*3, (self.params.score_box[3]-self.params.score_box[1])*3)
            # cv2.imshow('score', cv2.cvtColor(score_im, cv2.COLOR_GRAY2BGR))
            cv2.waitKey(1)
        #print("Played one move: {}, reaching terminal? {}".format(chosen_action, terminal))
        return screenshot, terminal
        
    # only called to start the game
    def start_tap(self):
        output_processor.tap((self.__denormalize_screen_position(self.params.restart_tap_position)))
        
    def restart(self, play=True):
        #print('>>>>>>> RESTART')
        tap = np.zeros(2)
        tap[1] = 1
        do_nothing = np.zeros(2)
        do_nothing[0] = 1
        time.sleep(0.5)
        _,terminal = self.frame_step(do_nothing)
        while terminal:
            output_processor.tap((self.__denormalize_screen_position(self.params.restart_tap_position)))
            _,terminal = self.frame_step(do_nothing) # just to grab if it's terminal state
            time.sleep(0.25)
        # if play:
        #   #print("TAP")
        #   time.sleep(0.7)
        #   self.frame_step(tap)
        # time.sleep(0.15)
        # output_processor.tap(self.__denormalize_screen_position(self.params.restart_tap_position))
        # time.sleep(0.3)
        # output_processor.tap(self.__denormalize_screen_position(self.params.restart_tap_position))
       
    # for ending transaction collection and play stage    
    def reach_terminal_state(self):
      do_nothing = np.zeros(2)
      do_nothing[0] = 1
      tap = np.zeros(2)
      tap[1] = 1
      
      print("Try to Reach terminal State")
      time.sleep(1)
      _,terminal = self.frame_step(do_nothing)
      
      while (not terminal):
        _,terminal = self.frame_step(tap)
        time.sleep(2.0) # make sure it's really picking up terminated
        _,terminal = self.frame_step(do_nothing)
        _,terminal = self.frame_step(do_nothing)
        _,terminal = self.frame_step(do_nothing)
      # input()
      # self.restart(play=False)

    # For this new trial version, this function is not used
    def __check_scoring(self, screenshot):
        check = True
        #print("ever got invoked?")
        for pixel, colour in zip(self.params.scoring_pixel_position, self.params.scoring_pixel_color):
            current_pixel_color = self.get_pixel_color(screenshot, pixel)
            #print("current pixel color:{} terminal pixel color: {} threshold: {}".format(current_pixel_color, colour, self.tolerance))
            if current_pixel_color < colour-self.tolerance or self.get_pixel_color(screenshot, pixel) > colour+self.tolerance:
                check = False
        return  check

    def __check_terminal_state(self, screenshot):
        check = True
        #print("ever got invoked?")
        for pixel, colour in zip(self.params.terminal_pixel_position, self.params.terminal_pixel_color):
            current_pixel_color = self.get_pixel_color(screenshot, pixel)
            #print("current pixel color:{} terminal pixel color: {} threshold: {}".format(current_pixel_color, colour, self.tolerance))
            if current_pixel_color < colour-self.tolerance or self.get_pixel_color(screenshot, pixel) > colour+self.tolerance:
                check = False
        return  check

    def __denormalize_screen_position(self, position):
        x = int(position[0] * self.emulator_resolution[0] + self.corner[0])
        y = int(position[1] * self.emulator_resolution[1] + self.corner[1])
        return x, y

    @staticmethod
    def denormalize_screenshot_position(screenshot, position):
        #print("screenshot: {}; position: {}".format(screenshot, position))
        x = int(position[0] * screenshot.shape[0])
        y = int(position[1] * screenshot.shape[1])
        return x, y

    @staticmethod
    def get_pixel_color(screenshot, position):
        x, y = Game.denormalize_screenshot_position(screenshot, position)
        return screenshot[x, y]
