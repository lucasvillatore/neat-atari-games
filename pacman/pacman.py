import neat

class Pacman:
    def __init__(self, game='pacman'):
        self.name = 'ALE/MsPacman-v5'
        self.frameskip = 1

    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'pacman/configs/neat-config'
        )

        return neat_configuration

    def run_game(self, env, network):
        return {} #todo

    def calculate_fitness(self, game_information):
        return 0 #todo