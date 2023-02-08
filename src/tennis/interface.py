import numpy as np
import math

class Tennis():
    def __init__(self, folder, net = None, checkpoint = None):
        self.folder = folder
        self.name = 'Tennis-v4'
        self.neat_config_path = f"{self.folder}/neat-config"
        self.net = net
        self.checkpoint = checkpoint
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

    def calculate_fitness(self, info, reward):

        player_x, player_y = self.get_player_coordinates(info)

        if abs(player_x - int(info['labels']['ball_x'])) < 15:
            reward += 1 * 0.001

        if abs(player_y - int(info['labels']['ball_y'])) < 15:
            reward += 1 * 0.001

        return reward

    def get_distance(self, player_x, player_y, ball_x, ball_y):
        distance = math.sqrt(math.pow(player_x - ball_x, 2) + math.pow(player_y - ball_y, 2)) 
        
        return distance

    def get_player_coordinates(self, info):
        if info['labels']['player_field'] == 0: # troca de campo
            return int(info['labels']['player_x']), int(info['labels']['player_y'])
        return int(info['labels']['enemy_x']), int(info['labels']['enemy_y'])

    def get_action(self, observation_space, net, step, info):

        is_first_action = step == 0


        if is_first_action:
            return 1

        player_x, player_y = self.get_player_coordinates(info)

        ball_is_on_right = 0 if int(info['labels']['ball_x']) > player_x else 1

        input_net = [
            player_x, 
            player_y,
            int(info['labels']['ball_x']),
            int(info['labels']['ball_y'])
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

        ball_direction = 0
        rebater = 0
        stagnation = 0
        current_y = 142
        tempo_mesmo_y = 0

        for current_step in range(steps):
            if current_step % 10 == 0:
                action = 1
            else:
                action = self.get_action(observation_space, net, current_step, game_information)

            observation_space, current_reward, done, game_information = env.step(action)

            player_x, player_y = self.get_player_coordinates(game_information)

            if current_y == player_y:
                stagnation += 1
            else:
                current_y = player_y
                stagnation = 0
            
            if stagnation >= 10:
                tempo_mesmo_y += 1

            if game_information['labels']['ball_direction'] != ball_direction:
                rebater += 1
                ball_direction = game_information['labels']['ball_direction']
            
            total_reward += self.calculate_fitness(game_information, current_reward)

            if done:
                break
        
        total_reward += rebater + game_information['frame_number'] * 0.0001 - tempo_mesmo_y * 0.05

        if total_reward < 0:
            total_reward = 0

        return total_reward