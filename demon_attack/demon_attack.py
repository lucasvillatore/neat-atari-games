import neat

class DemonAttack:
    def __init__(self, game='DemonAttack'):
        self.name = 'ALE/DemonAttack-ram-v5'
        self.frameskip = 1

    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'demon_attack/configs/neat-config'
        )

        return neat_configuration