from math import sqrt
from time import sleep
import neat
import numpy as np
from random import randrange
import cv2
import imutils

class Pong:
    def __init__(self, game='Pong'):
        self.name = 'ALE/Pong-v5'
        self.frameskip = 1
        self.my_rgb = None
        self.ball_rgb = [236, 236, 236]


    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'pong/configs/neat-config'
        )

        return neat_configuration

    def run_game(self, env, network):

        frames = 0 
        n_state = env.reset(seed=randrange(10000))

        done = False
        fitness = 0
        while -10 < fitness < 10:
            frames += 1

            if frames < 60:
                n_state, reward, done, info = env.step(0)
                continue
            inputs = []

            
            my_coordinates = self.get_my_position(n_state)
            ball_coordinates = self.get_ball_posittion(n_state)
            distance = sqrt((my_coordinates[0] - ball_coordinates[0])** 2 + (my_coordinates[1] - ball_coordinates[1])** 2)
            
            inputs = [my_coordinates[1]] + [ball_coordinates[1]] + [distance]
            
            ai_decision = network.activate(inputs)
            action = np.argmax(ai_decision)

            n_state, reward, done, info = env.step(action)   
            fitness += reward

        env.close()

        return fitness 

    def get_my_position(self, state):
        if self.my_rgb is None:
            self.set_my_rgb(state)

        state = state[35:190, 0:160]
        my_position_rgb = np.array(self.my_rgb)
        my_position_state = cv2.inRange(state, my_position_rgb, my_position_rgb)

        coordinates = self.get_coordinates(my_position_state)
        
        if len(coordinates) > 0:
            return coordinates[0]

        return [0,0]

    def set_my_rgb(self, state):
        self.my_rgb = state[105][143]

    def get_ball_posittion(self, state):
        tmp = state[35:190, 0:160]

        ball_position_rgb = np.array(self.ball_rgb)
        ball_position_state = cv2.inRange(tmp, ball_position_rgb, ball_position_rgb)

        coordinates = self.get_coordinates(ball_position_state)

        if len(coordinates) > 0:
            return coordinates[0]
        return [0,0]

    def calculate_fitness(self, game_information):
        return game_information #todo

    def get_coordinates(self, state):
        blurred = cv2.GaussianBlur(state, (5, 5), 0)
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        coordinates = []
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coordinates.append([cX,cY+35])
            except Exception as err:
                pass

        return coordinates