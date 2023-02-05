import numpy as np
import math
import random

class Breakout():
    def __init__(self, folder, net = None, checkpoint = None):
        self.folder = folder
        self.name = 'Breakout-v4'
        self.neat_config_path = f"{self.folder}/neat-config"
        self.net = net
        self.checkpoint = checkpoint
        self.node_names = {
            -1 : "ball_direction", 
            0 : "NOOP",
            1 : "FIRE",
            2 : "UP",
            3 : "RIGHT",
            4 : "LEFT",
            5 : "DOWN",
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

    def calculate_fitness(self, info, reward):

        # player_y = 180
        if abs(int(info['labels']['player_x']) - int(info['labels']['ball_x'])) < 10:
            reward += 1 * 0.03

        return reward

    def get_action(self, observation_space, net, step, info):
        is_first_action = step == 0

        if is_first_action:
            return 1

        ball_is_on_right = 0 if int(info['labels']['ball_x']) > int(info['labels']['player_x']) else 1

        input_net = [
            ball_is_on_right,
        ]

        try:
            output = net.activate(input_net)
            action = np.argmax(output)
        except Exception as err:
            action = 1
        
        return action
        
    def run(self, net, env, steps):

        observation_space = env.reset()
        game_information = {}
        total_reward = 0.0

        action = 1        
        for current_step in range(steps):

            action = self.get_action(observation_space, net, current_step, game_information)

            if current_step % 30 == 0:
                action = 1
                
            observation_space, current_reward, done, game_information = env.step(action)
            total_reward += self.calculate_fitness(game_information, current_reward)

            
            if done:
                break
        
        total_reward += game_information['labels']['blocks_hit_count'] * 0.5 + game_information['frame_number'] * 0.0001

        return total_reward