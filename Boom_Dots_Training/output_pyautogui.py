import pyautogui


class OutputPyautogui:

    def __init__(self):
        pass

    def tap(self, point):
        pyautogui.click(point[0], point[1])