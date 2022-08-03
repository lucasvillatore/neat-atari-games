import neat

class NeatTraining:
    def __init__(self, parameters = None, configuration=None):
        self.max_generations = parameters.generations
        self.actual_generation = 0
        self.configuration = configuration
        
        self.neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'configs/neat-config'
        )
        
    def get_action(self):
        pass
