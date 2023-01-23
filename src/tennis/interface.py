import numpy as np
import math

class Tennis():
    def __init__(self, net = None, checkpoint = None):
        self.actions = {}
        self.name = 'Tennis-v4'
        self.neat_config_path = "./tennis/configs/neat-config"
        self.folder = "./tennis"
        self.net = net
        self.checkpoint = checkpoint

    def calculate_fitness(self, info, reward):
        if  self.get_distance(
            int(info['labels']['player_x']),
            int(info['labels']['player_y']),
            int(info['labels']['ball_x']),
            int(info['labels']['ball_y']),
        ) < 20:
            return 0.05 + reward

        return reward

    def get_distance(self, player_x, player_y, ball_x, ball_y):
        distance = math.sqrt(math.pow(player_x - ball_x, 2) + math.pow(player_y - ball_y, 2)) 
        
        return distance
    def get_action(self, observation_space, net, step, info):

        is_first_action = step == 0
        
        if is_first_action:
            return 1

        my_player_distance_to_ball = self.get_distance(
            int(info['labels']['player_x']),
            int(info['labels']['player_y']),
            int(info['labels']['ball_x']),
            int(info['labels']['ball_y']),
        )
        ball_is_on_left = 1 if int(info['labels']['ball_y']) > int(info['labels']['player_y']) else 0
        input_net = [
            my_player_distance_to_ball,
            ball_is_on_left
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
            if current_step % 30 == 0:
                action = 1
            else:
                action = self.get_action(observation_space, net, current_step, game_information)

            observation_space, current_reward, done, game_information = env.step(action)

            total_reward += self.calculate_fitness(game_information, current_reward)

            if done:
                break

        
        return total_reward