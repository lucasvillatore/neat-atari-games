import neat

class FlappyBird:
    def __init__(self, game='FlappyBird'):
        self.name = 'FlappyBird-ram-v0'
        self.frameskip = 1

    def get_neat_configuration(self):
        neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'flappy_bird/configs/neat-config'
        )

        return neat_configuration
