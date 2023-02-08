import numpy as np
import math
import random
import time

class Pong():
    def __init__(self, folder, net = None, checkpoint = None):
        self.folder = folder
        self.neat_config_path = f"{self.folder}/neat-config"
        self.net = net
        self.checkpoint = checkpoint
        self.name = 'Pong-v4'
        self.node_names = {
            -1 : "player_x", 
            -2 : "player_y", 
            -3 : "ball_x", 
            -4 : "ball_y", 
            0 : "NOOP",
            1 : "FIRE",
            2 : "RIGHT",
            3 : "LEFT",
            4 : "RIGHTFIRE",
            5 : "LEFTRIGHTFIRE",
            6 : "UPRIGHT",
            7 : "UPLEFT",
            8 : "DOWNRIGHT",
            9 : "DOWNLEFT",
            10 : "UPFIRE",
            11 : "RIGHTFIRE",
            12 : "LEFTFIRE",
            13 : "DOWNFIRE",
            14 : "UPRIGHTFIRE",
            15 : "UPLEFTFIRE",
            16 : "DOWNRIGHTFIRE",
            17 : "DOWNLEFTFIRE",
        }

    def calculate_fitness(self, info, reward, last_ball_direction):
        ball_direction = int(info['labels']['ball_direction'])

        if ball_direction < 10: # se conseguiu rebater
            reward += 1 * 0.05

        if abs(int(info['labels']['player_y']) - int(info['labels']['ball_y'])) < 10:
            reward += 1 * 0.05

        return reward

    def get_action(self, observation_space, net, step, info):

        is_first_action = step == 0

        if is_first_action:
            return 0

        ball_is_upper = 1 if int(info['labels']['ball_y']) > int(info['labels']['player_y']) else 0
        
        input_net = [
            int(info['labels']['player_x']),
            int(info['labels']['player_y']),
            int(info['labels']['ball_x']),
            int(info['labels']['ball_y']),
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


        for current_step in range(steps):
            action = self.get_action(observation_space, net, current_step, game_information)

            observation_space, current_reward, done, game_information = env.step(action)

            last_ball_direction = game_information['labels']['ball_direction']

            total_reward += self.calculate_fitness(game_information, current_reward, last_ball_direction)

            if done:
                break

        total_reward += game_information['labels']['enemy_score'] * -1 * 2 + game_information['labels']['player_score'] * 2
        total_reward += game_information['frame_number'] * 0.0001

        if total_reward < 0:
            total_reward = 0

        return total_reward

