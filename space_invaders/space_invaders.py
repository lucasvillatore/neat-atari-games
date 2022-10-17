import neat

class SpaceInvaders:
    def __init__(self, game='SpaceInvaders'):
        self.name = 'ALE/SpaceInvaders-ram-v0'
        self.frameskip = 1

    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'space_invaders/configs/neat-config'
        )

        return neat_configuration
