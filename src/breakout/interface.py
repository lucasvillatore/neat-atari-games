import numpy as np
import math
import random

class Breakout():
    def __init__(self, net = None, checkpoint = None):
        self.actions = {0: 'NOOP', 1: 'FIRE', 2: 'LEFT', 3: 'RIGHT'}
        self.name = 'Breakout-v4'
        self.neat_config_path = "./breakout/configs/neat-config"
        self.folder = "./breakout"
        self.net = net
        self.checkpoint = checkpoint

    def get_distance(self, player_x, player_y, ball_x, ball_y):
        distance = math.sqrt(math.pow(player_x - ball_x, 2) + math.pow(player_y - ball_y, 2)) 
        
        return distance

    def calculate_fitness(self, info, reward):

        if abs(int(info['labels']['player_x']) - int(info['labels']['ball_x'])) < 10:
            reward += 1 * 0.03
        
        if abs(int(info['labels']['player_y']) - 32 - int(info['labels']['ball_y'])) < 10:
            reward += 1 * 0.05

        return reward

    def get_action(self, observation_space, net, step, info):
        is_first_action = step == 0

        if is_first_action:
            return 1


        # ball_is_on_left = 0 if int(info['labels']['ball_x']) > int(info['labels']['player_x']) else 1

        # my_player_distance_to_ball = self.get_distance(
        #     int(info['labels']['player_x']),
        #     int(info['labels']['player_y']) - 32,
        #     int(info['labels']['ball_x']),
        #     int(info['labels']['ball_y']),
        # )

        input_net = [
            int(info['labels']['player_x']),
            int(info['labels']['player_y']) - 32,
            int(info['labels']['ball_x']),
            int(info['labels']['ball_y']),
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