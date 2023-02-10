import numpy as np
import math
import random
import time

class PongXY():
    def get_inputs(self, info):
        return [
            int(info['labels']['player_x']),
            int(info['labels']['player_y']),
            int(info['labels']['ball_x']),
            int(info['labels']['ball_y']),
        ]

    def get_action(self, action):
        return action

    
    def get_node_names(self, full_action_space):

        node_names = {-1 : "player_x", -2 : "player_y", -3 : "ball_x", -4 : "ball_y"}

        node_names[0] = "NOOP"
        node_names[1] = "FIRE"
        node_names[2] = "RIGHT"
        node_names[3] = "LEFT"
        node_names[4] = "RIGHTFIRE"
        node_names[5] = "LEFTFIRE"

        if not full_action_space:
            return node_names
        
        node_names[0] = "NOOP"
        node_names[1] = "FIRE"
        node_names[2] = "RIGHT"
        node_names[3] = "LEFT"
        node_names[4] = "RIGHTFIRE"
        node_names[5] = "LEFTRIGHTFIRE"
        node_names[6] = "UPRIGHT"
        node_names[7] = "UPLEFT"
        node_names[8] = "DOWNRIGHT"
        node_names[9] = "DOWNLEFT"
        node_names[10] = "UPFIRE"
        node_names[11] = "RIGHTFIRE"
        node_names[12] = "LEFTFIRE"
        node_names[13] = "DOWNFIRE"
        node_names[14] = "UPRIGHTFIRE"
        node_names[15] = "UPLEFTFIRE"
        node_names[16] = "DOWNRIGHTFIRE"
        node_names[17] = "DOWNLEFTFIRE"

        return node_names

class PongY():
    def get_inputs(self, info):
        value = 8
        if int(info['labels']['player_y']) - value > int(info['labels']['ball_y']) > int(info['labels']['player_y']) + value:
            ball_direction = 1
        elif int(info['labels']['ball_y'] + value > int(info['labels']['player_y'])): 
            ball_direction = 2
        else:
            ball_direction = 0

        return [ball_direction]

    def get_action(self, action):
        return action

    def get_node_names(self, full_action_space):

        node_names = {-1 : "ball_direction"}
        node_names[0] = "NOOP"
        node_names[1] = "FIRE"
        node_names[2] = "RIGHT"
        node_names[3] = "LEFT"
        node_names[4] = "RIGHTFIRE"
        node_names[5] = "LEFTFIRE"

        if not full_action_space:
            return node_names
        
        node_names[0] = "NOOP"
        node_names[1] = "FIRE"
        node_names[2] = "RIGHT"
        node_names[3] = "LEFT"
        node_names[4] = "RIGHTFIRE"
        node_names[5] = "LEFTRIGHTFIRE"
        node_names[6] = "UPRIGHT"
        node_names[7] = "UPLEFT"
        node_names[8] = "DOWNRIGHT"
        node_names[9] = "DOWNLEFT"
        node_names[10] = "UPFIRE"
        node_names[11] = "RIGHTFIRE"
        node_names[12] = "LEFTFIRE"
        node_names[13] = "DOWNFIRE"
        node_names[14] = "UPRIGHTFIRE"
        node_names[15] = "UPLEFTFIRE"
        node_names[16] = "DOWNRIGHTFIRE"
        node_names[17] = "DOWNLEFTFIRE"

        return node_names

class Pong():
    def __init__(self, folder, tmp, full_action_space = False, net = None, checkpoint = None):
        self.full_action_space = full_action_space
        self.folder = folder
        self.neat_config_path = f"{self.folder}/neat-config"
        self.net = net
        self.checkpoint = checkpoint
        self.name = 'Pong-v4'
        self.tmp = tmp
        self.node_names = self.tmp.get_node_names(full_action_space) 

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

        input_net = self.tmp.get_inputs(info)

        try:
            output = net.activate(input_net)
            action = np.argmax(output)
        except Exception as err:
            action = 0

        return self.tmp.get_action(action)

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

