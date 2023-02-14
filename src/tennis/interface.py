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

class TennisX:  
    def get_inputs(self, info, ball_out):
        ball_x = int(info['labels']['ball_x'])
        ball_y = int(info['labels']['ball_y'])
        player_x = int(info['labels']['player_x'])

        if ball_x <= player_x: 
            ball_direction = 1
        else:
            ball_direction = 0
        
        if (ball_x == 77 and ball_y == 142) or (ball_x == 119 and ball_y == 7) or (ball_x == 2 and ball_y == 161): 
            ball_direction = -1
            ball_out = ball_out

        return [ball_direction, ball_out]

    def get_action(self, action):
        return action

    def get_node_names(self, full_action_space):
        node_names = {-1 : "ball_left", -2: "ball_out"}


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

    def calculate_fitness(self, info):

        player_x, player_y = self.get_player_coordinates(info)

        reward = 0
        ball_x = int(info['labels']['ball_x'])
        ball_y = int(info['labels']['ball_y'])

        if (ball_x == 77 and ball_y == 142) or (ball_x == 119 and ball_y == 7):
            return 0
        
        if abs(player_x - ball_x) < 15:
            reward += 1 * 0.003

        return reward

    def get_distance(self, player_x, player_y, ball_x, ball_y):
        distance = math.sqrt(math.pow(player_x - ball_x, 2) + math.pow(player_y - ball_y, 2)) 
        
        return distance

    def get_player_coordinates(self, info):
        if info['labels']['player_field'] == 0: # troca de campo
            return int(info['labels']['player_x']), int(info['labels']['player_y'])
        return int(info['labels']['enemy_x']), int(info['labels']['enemy_y'])

    def get_action(self, net, step, info, ball_out):

        is_first_action = step == 0

        if is_first_action:
            return 1

        input_net = self.tmp.get_inputs(info, ball_out)

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

        ball_out = 0
        in_game = 0
        for current_step in range(steps):
            action = self.get_action(net, current_step, game_information, ball_out)

            observation_space, current_reward, done, game_information = env.step(action)

            ball_x = int(game_information['labels']['ball_x'])
            ball_y = int(game_information['labels']['ball_y'])

            if (ball_x == 77 and ball_y == 142) or (ball_x == 119 and ball_y == 7) or (ball_x == 2 and ball_y == 161):
                ball_out += 1
            else:
                in_game += 1
                ball_out = 0

                if game_information['labels']['ball_direction'] != ball_direction:
                    rebater += 1  * 0.03
                    ball_direction = game_information['labels']['ball_direction']
                
                total_reward += self.calculate_fitness(game_information)


            if done:
                break
        
        # print()
        # print()
        # print("reward rebater {}".format(rebater))
        # print("reward aproximacao x {}".format(total_reward))
        # print("reward in game {}".format(in_game * 0.05 ))
        total_reward += rebater + in_game * 0.05

        if total_reward < 0:
            total_reward = 0

        return total_reward