from basic_interface import InterfaceGames
import numpy as np

class Alien(InterfaceGames):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = {}

    def calculate_fitness(self, info, reward):
        return reward

    def get_action(self, observation_space, net, step, info):

        is_first_action = step == 0
        
        if is_first_action:
            return 0

        input_net = []

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
        