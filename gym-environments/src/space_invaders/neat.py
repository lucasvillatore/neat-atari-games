import neat

class NeatTraining:
    def __init__(self, parameters = None, configuration=None):
        self.max_generations = parameters.generations
        self.actual_generation = 0
        self.configuration = configuration
        self.neat = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpecieSet,
            neat.DefaultStagnation,
            './configs/config-neat'
        )
    def get_action(self):
        pass
