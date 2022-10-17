import neat

class Seaquest:
    def __init__(self, game='Seaquest'):
        self.name = 'ALE/Seaquest-ram-v0'
        self.frameskip = 1

    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'seaquest/configs/neat-config'
        )

        return neat_configuration
