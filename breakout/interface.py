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
        x, y, c = run_env.observation_space.shape

        x = int(x/10)
        y = int(y/10)
        ob = cv.resize(image, (x, y))
        ob = cv.cvtColor(ob, cv.COLOR_BGR2GRAY)
        flatten = ob.flatten()

        try:
            outputs = net.activate(flatten)
            action = np.argmax(outputs)
        except Exception as err:
            action = 1

        return run_env.step(action)

    def draw_net(self, game, winner):
        node_names = {0: 'NOOP', 1: 'FIRE', 2: 'LEFT', 3: 'RIGHT'}
        visualize.draw_net(game.config, winner, True, node_names=node_names)
        visualize.draw_net(game.config, winner, True, node_names=node_names, prune_unused=True)