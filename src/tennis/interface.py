import numpy as np
import math


class TennisXY:
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

class TennisPongXY:
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

class TennisX:
    def get_inputs(self, info):
        value = 8
        if int(info['labels']['player_x']) - value > int(info['labels']['ball_x']) > int(info['labels']['player_x']) + value:
            ball_direction = 1
        elif int(info['labels']['ball_x'] + value > int(info['labels']['player_x'])): 
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

class TennisPongY:
    def get_inputs(self, info):
        value = 8
        if int(info['labels']['player_x']) - value > int(info['labels']['ball_x']) > int(info['labels']['player_x']) + value:
            ball_direction = 1
        elif int(info['labels']['ball_x'] + value > int(info['labels']['player_x'])): 
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

class Tennis():
    def __init__(self,  folder, tmp, full_action_space = False, net = None, checkpoint = None):
        self.full_action_space = full_action_space
        self.folder = folder
        self.name = 'Tennis-v4'
        self.neat_config_path = f"{self.folder}/neat-config"
        self.net = net
        self.tmp = tmp
        self.checkpoint = checkpoint
        self.node_names = self.tmp.get_node_names(self.full_action_space)

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