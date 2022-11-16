import pickle
from neat import statistics


class InterfaceGames:

    def __init__(self, name, config, folder, checkpoint = None) -> None:
        self.name = name
        self.config = config
        self.folder = folder
        self.checkpoint = checkpoint

    def save_checkpoint(self, population):
        filename = '/checkpoint'
        if self.checkpoint is not None:
            filename += '-using-checkpoint'

        population.save_checkpoint(self.folder + filename)

    def save_statistics(self, population):
        fitness_history = '/fitness_history.csv'
        speciation_history = "/speciation.csv"
        species_fitness = "/species_fitness.csv"
        
        if self.checkpoint is not None:
            fitness_history = '/fitness_history-using-checkpoint.csv'
            speciation_history = '/speciation-using-checkpoint.csv'
            species_fitness = '/species_fitness-using-checkpoint.csv'

        statistics.save_stats(population.statistics, filename=self.folder + fitness_history)
        statistics.save_species_count(population.statistics, filename=self.folder + speciation_history)
        statistics.save_species_fitness(population.statistics, filename=self.folder + species_fitness)

    def save_winner(self, population):
        winner = population.statistics.best_genome()

        filename = '/winner.pkl'

        if self.checkpoint is not None:
            filename = '/winner-using-checkpoint.pkl'

        with open(self.folder + filename, 'wb') as file:
            pickle.dump(winner, file, 1)