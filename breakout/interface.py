from basic_interface import InterfaceGames
import cv2 as cv
import numpy as np
import random

kernel = np.ones((2,2), np.uint8) 
class Breakout(InterfaceGames):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_fitness(self, reward):

        return reward

    def run_step(self, image, net, run_env):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        _, bw_img = cv.threshold(gray_image, 40, 255, cv.THRESH_BINARY)

        tmp = cv.dilate(bw_img, kernel, iterations=1)
        running_game = tmp.copy()

        try:
            outputs = net.serial_activate(running_game)
            action = np.argmax(outputs)
        except Exception as err:
            action = 1

        return run_env.step(action)

