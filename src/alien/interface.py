from basic_interface import InterfaceGames

class Alien(InterfaceGames):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def calculate_fitness(self, reward):

        return reward

    