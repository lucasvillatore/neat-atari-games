import numpy as np
import math
import random
import time
import cv2

class Skiing():
    def __init__(self, folder, net = None, checkpoint = None):
        self.folder = folder
        self.name = 'Skiing-v4'
        self.neat_config_path = f"{self.folder}/neat-config"
        self.net = net
        self.checkpoint = checkpoint
        self.node_names = {-1 : "flag_is_on_right", 0: "noop", 1 : "right", 2: "left"}

    def calculate_fitness(self, info, reward, observation_space):

        flag_x, flag_y = self.get_flag_coordinates(observation_space)

        if reward < 0:
            reward = 0

        # print(flag_x, flag_y)
        # time.sleep(0.3)
        if flag_x > int(info['labels']['player_x']) >= flag_x + 20 and 155 > flag_y > 170 :
            reward += 1

        return reward

    def get_flag_coordinates(self, observation_space):
        index = 0
        for i in range(80, 86):
            if observation_space[i] == 0:
                flag_x = int(observation_space[64 + index])
                flag_y = int(observation_space[87 + index])

                return flag_x, flag_y
            else:
                index += 1
                flag_x = 0
                flag_y = 0
        
        return flag_x, flag_y


    def get_action(self, observation_space, net, step, info, is_stagned):

        is_first_action = step == 0
        
        if is_first_action:
            return 0

        flag_x, flag_y = self.get_flag_coordinates(observation_space)

        flag_is_on_right = 0 if flag_x + 10 > int(info['labels']['player_x']) else 1
        # print(flag_is_on_right)
        # time.sleep(0.3)

        input_net = [
            flag_is_on_right,
            #is_stagned
        ]
        try:
            output = net.activate(input_net)
            action = np.argmax(output)
        except Exception as err:
            action = 0
        
        # return 0
        # return action

        if action in (2, 4):
            return 1
        if action in (3, 5):
            return 2

        return 0
        
    def run(self, net, env, steps):

        observation_space = env.reset()
        game_information = {}
        total_reward = 0.0

        action = 0

        stagnation = 0
        tempo_mesmo_y = 0
        mesmo_obj = observation_space[80]
        for current_step in range(steps):

            action = self.get_action(observation_space, net, current_step, game_information, tempo_mesmo_y)

            observation_space, current_reward, done, game_information = env.step(action)

            if observation_space[80] == mesmo_obj:
                stagnation += 1
            else:
                mesmo_obj = observation_space[80]
                stagnation = 0
                tempo_mesmo_y = 0

            if stagnation >= 20:
                tempo_mesmo_y = 1

            total_reward += self.calculate_fitness(game_information, current_reward, observation_space)

            if done:
                break
        # total_reward += - tempo_mesmo_y * 0.05
        
        # if total_reward < 0:
        #     total_reward = 0
        return total_reward