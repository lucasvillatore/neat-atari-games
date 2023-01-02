from basic_interface import InterfaceGames
import cv2 as cv
import numpy as np
import random
from common import visualize

kernel = np.ones((2,2), np.uint8) 
class Breakout(InterfaceGames):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_fitness(self, reward):

        return reward

    def run_step(self, image, net, run_env):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        _, bw_img = cv.threshold(gray_image, 40, 255, cv.THRESH_BINARY)

        running_game = cv.dilate(bw_img, kernel, iterations=1)
        try:
            outputs = net.activate(running_game)
            action = np.argmax(outputs)
        except Exception as err:
            action = 1

        return run_env.step(action)

    def draw_net(self, game, winner):
        node_names = {0: 'NOOP', 1: 'FIRE', 2: 'LEFT', 3: 'RIGHT'}
        visualize.draw_net(game.config, winner, True, node_names=node_names)
        visualize.draw_net(game.config, winner, True, node_names=node_names, prune_unused=True)