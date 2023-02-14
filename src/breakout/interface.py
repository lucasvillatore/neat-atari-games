import numpy as np

class BreakoutXY():
    def get_inputs(self, info, ball_out):
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

class BreakoutX():
    def ball_is_on_game(self, info):
        return not int(info['labels']['ball_y']) == 0

    def get_inputs(self, info, ball_out):
        ball_y = int(info['labels']['ball_y'])
        ball_x = int(info['labels']['ball_x'])
        player_x = int(info['labels']['player_x'])

        if ball_x <= player_x: 
            ball_left = 1
            ball_right = 0
            ball_out = ball_out
        else:
            ball_left = 0
            ball_right = 1
            ball_out = ball_out
        
        if ball_y == 0:
            ball_left = -1
            ball_right = -1
            ball_out = ball_out

        return [ball_left, ball_right, ball_out]
        

    def get_action(self, action):
        return action

    def get_node_names(self, full_action_space):

        node_names = {-1 : "ball_left", -2 : "ball_right", -3 : "ball_out_field"}

        node_names[0] = "NOOP"
        node_names[1] = "FIRE"
        node_names[2] = "RIGHT"
        node_names[3] = "LEFT"

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

class Breakout():
    def __init__(self, folder, tmp, full_action_space = False, net = None, checkpoint = None):
        self.full_action_space = full_action_space
        self.folder = folder
        self.name = 'Breakout-v4'
        self.neat_config_path = f"{self.folder}/neat-config"
        self.net = net
        self.checkpoint = checkpoint
        self.tmp = tmp
        self.node_names = self.tmp.get_node_names(full_action_space) 

    def ball_is_on_game(self, info):
        return not int(info['labels']['ball_y']) == 0

    def calculate_fitness(self, info, reward):

        reward = 0

        if abs(int(info['labels']['player_x']) - int(info['labels']['ball_x'])) < 15 and self.ball_is_on_game(info):
            reward += 1 * 0.03

        
        return reward

    def get_action(self, net, step, info, ball_out):
        is_first_action = step == 0

        if is_first_action:
            return 1

        input_net = self.tmp.get_inputs(info, ball_out)
        
        try:
            output = net.activate(input_net)
            action = np.argmax(output)
        except Exception as err:
            action = 1
        
        return self.tmp.get_action(action)
        
    def run(self, net, env, steps):

        observation_space = env.reset()
        game_information = {}
        total_reward = 0.0

        action = 1      
        out_of_game = 0
        in_game = 0
        ball_hits = 0
        ball_out = 0
        for current_step in range(steps):
            
            action = self.get_action(net, current_step, game_information, ball_out)

            observation_space, current_reward, done, game_information = env.step(action)
            ball_hits += self.calculate_fitness(game_information, current_reward)
            
            if not self.ball_is_on_game(game_information):
                ball_out += 1
                out_of_game += 1
            else:
                ball_out = 0
                in_game += 1
        
            if done:
                break
        
        all_points = in_game + out_of_game

        # total_reward = - (all_points - out_of_game) * 0.01 + (all_points - in_game) * 0.005 + int(game_information['labels']['blocks_hit_count']) * 0.5
        # print()
        # print()
        # print("reward rebatida {}".format(ball_hits))
        # print("reward blocks {}".format( int(game_information['labels']['blocks_hit_count']) * 0.5 ))
        # print("reward in game {}".format( (all_points - in_game) * 0.005) )
        # print("reward out of game -{}".format(  - (all_points - out_of_game) * 0.01 ))
        total_reward += ball_hits + in_game * 0.05 + int(game_information['labels']['blocks_hit_count']) * 0.5

        if total_reward < 0:
            return 0


        return total_reward