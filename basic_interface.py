import pickle
from neat import statistics


class InterfaceGames:

    def __init__(self, name, config, folder) -> None:
        self.name = name
        self.config = config
        self.folder = folder

    def save_checkpoint(self, population):
        population.save_checkpoint(self.folder + "/checkpoint")

    def save_statistics(self, population):
        statistics.save_stats(population.statistics, filename=self.folder + "/fitness_history.csv")
        statistics.save_species_count(population.statistics, filename=self.folder + "/speciation.csv")
        statistics.save_species_fitness(population.statistics, filename=self.folder + "/species_fitness.csv")

    def save_winner(self, population):
        winner = population.statistics.best_genome()
    
        with open(self.folder + '/winner.pkl', 'wb') as file:
            pickle.dump(winner, file, 1)