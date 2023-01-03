from basic_interface import InterfaceGames
import cv2 as cv
import numpy as np
import random
import math
from common import visualize

kernel = np.ones((2,2), np.uint8) 
class Breakout(InterfaceGames):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = {0: 0, 1: 0, 2: 0, 3: 0}

    def calculate_fitness(self, info):


        if abs(info['labels']['ball_x'] - info['labels']['player_x']) < 5:
            return 2

        return 0

    def run_step(self, image, net, run_env, step, info):

        if step == 0 or step % 3 != 0:
            return 1
        

        a = (info['labels']['ball_x'] - info['labels']['player_x'])**2
        b = (info['labels']['ball_y'] - 190)**2
        
        distancia = int(math.sqrt(a + b))

        tmp = [
            distancia,
            info['labels']['player_x'],
            info['labels']['ball_x'],
            info['labels']['ball_y']
        ]

        try:
            output = net.activate(tmp)
            action = np.argmax(output)
        except Exception as err:
            action = 1
        
        return action

    def draw_net(self, game, winner):
        node_names = {0: 'NOOP', 1: 'FIRE', 2: 'LEFT', 3: 'RIGHT'}
        visualize.draw_net(game.config, winner, True, node_names=node_names)
        visualize.draw_net(game.config, winner, True, node_names=node_names, prune_unused=True)