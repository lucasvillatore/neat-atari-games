import neat

class MontezumaRevenge:
    def __init__(self, game='MontezumaRevenge'):
        self.name = 'ALE/MontezumaRevenge-ram-v5'
        self.frameskip = 1

    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'montezuma_revenge/configs/neat-config'
        )

        return neat_configuration
