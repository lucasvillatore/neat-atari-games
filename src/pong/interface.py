import numpy as np
import math
import random
import time

class Pong():
    def __init__(self, net = None, checkpoint = None):
        self.actions = {}
        self.name = 'Pong-v4'
        self.neat_config_path = "./pong/configs/neat-config"
        self.folder = "./pong"
        self.net = net
        self.checkpoint = checkpoint

    def calculate_fitness(self, info, reward, last_ball_direction):
        ball_direction = int(info['labels']['ball_direction'])

        if ball_direction < 10: # se conseguiu rebater
            reward += 1 * 0.5

        if abs(int(info['labels']['player_x']) - int(info['labels']['ball_x'])) < 10:
            reward += 1 * 0.03

        if abs(int(info['labels']['player_y']) - int(info['labels']['ball_y'])) < 10:
            reward += 1 * 0.05

        return reward

    def get_distance(self, player_x, player_y, ball_x, ball_y):
        distance = math.sqrt(math.pow(player_x - ball_x, 2) + math.pow(player_y - ball_y, 2))

        return distance

    def get_action(self, observation_space, net, step, info):

        is_first_action = step == 0

        if is_first_action:
            return 0

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

        last_ball_direction = 0

        action = 0
        for current_step in range(steps):
            action = self.get_action(observation_space, net, current_step, game_information)

            observation_space, current_reward, done, game_information = env.step(action)

            last_ball_direction = game_information['labels']['ball_direction']

            total_reward += self.calculate_fitness(game_information, current_reward, last_ball_direction)

            if done:
                break

        total_reward += game_information['labels']['enemy_score'] * -1 * 5 + game_information['labels']['player_score'] * 2
        total_reward += game_information['frame_number'] * 0.0001

        if total_reward < 0:
            total_reward = 0
        return total_reward
