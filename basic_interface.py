import pickle
from neat import statistics


class InterfaceGames:

    def __init__(self, name, config, folder, checkpoint = None) -> None:
        self.name = name
        self.config = config
        self.folder = folder
        self.checkpoint = checkpoint

    def save_checkpoint(self, population):
        population.save_checkpoint(self.folder + "/checkpoint-using-checkpoint")

    def save_statistics(self, population):
        statistics.save_stats(population.statistics, filename=self.folder + "/fitness_history-using-checkpoint.csv")
        statistics.save_species_count(population.statistics, filename=self.folder + "/speciation-using-checkpoint.csv")
        statistics.save_species_fitness(population.statistics, filename=self.folder + "/species_fitness-using-checkpoint.csv")

    def save_winner(self, population):
        winner = population.statistics.best_genome()
    
        with open(self.folder + '/winner-using-checkpoint.pkl', 'wb') as file:
            pickle.dump(winner, file, 1)