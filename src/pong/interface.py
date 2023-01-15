from common.basic_interface import InterfaceGames
import numpy as np
import time
import cv2
import random
class Pong(InterfaceGames):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = {}

    def calculate_fitness(self, info, reward):
        
        # return reward
        if abs(int(info['labels']['ball_y']) - int(info['labels']['player_y'])) < 5:
            return 1 + reward
        return 0

    def get_action(self, observation_space, net, step, info):

        is_first_action = step == 0
        
        if is_first_action:
            return 0

        input_net = [
            info['labels']['player_x'],
            info['labels']['player_y'],
            info['labels']['ball_x'],
            info['labels']['ball_y'],
        ]

        try:
            output = net.activate(input_net)
            action = np.argmax(output)
        except Exception as err:
            action = 0
        
        return action
        
    def run(self, net, env, steps):

        observation_space = env.reset()
        game_information = {}
        total_reward = 0.0

        last_ball_direction = 0
        for current_step in range(steps):
            action = self.get_action(observation_space, net, current_step, game_information)
            observation_space, current_reward, done, game_information = env.step(action)
            
            ball_direction = game_information['labels']['ball_direction']

            if (last_ball_direction != 0 or ball_direction != 255) and ball_direction != last_ball_direction:
                current_reward = 5
            else:
                current_reward = 0

            last_ball_direction = ball_direction

            total_reward += self.calculate_fitness(game_information, current_reward)

            if observation_space[14] == 21:
                done = True
                total_reward = 5000
            if done:
                break
        
        return total_reward