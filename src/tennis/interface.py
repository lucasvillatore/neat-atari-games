import numpy as np

class Tennis():
    def __init__(self, net = None, checkpoint = None):
        self.actions = {}
        self.name = 'Tennis-v4'
        self.neat_config_path = "./tennis/configs/neat-config"
        self.folder = "./tennis"
        self.net = net
        self.checkpoint = checkpoint

    def calculate_fitness(self, info, reward):
        return reward


    def get_action(self, observation_space, net, step, info):

        is_first_action = step == 0
        
        if is_first_action:
            return 0

        input_net = observation_space

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

            action = self.get_action(observation_space, net, current_step, game_information)

            observation_space, current_reward, done, game_information = env.step(action)

            total_reward += self.calculate_fitness(game_information, current_reward)

            if done:
                break

        
        return total_reward