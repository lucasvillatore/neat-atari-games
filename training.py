import neat
import multiprocessing
import gym
import pickle
from pacman.pacman import Pacman

class Training:

    def __init__(self, game, generations):
        self.game = game
        self.generations = generations

    def train(self, genome, configuration):
        env = gym.make(self.game.name, frameskip=self.game.frameskip)
        neat_network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, configuration)

        game_information = self.game.run_game(env, neat_network)

        fitness = self.game.calculate_fitness(game_information)

        return fitness

    def save_winner(self, winner)
        with open('winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)

if __name__ == '__main__':
    game_training = Training(game=Pacman(), generations=100)
    config = game_training.game.get_neat_configuration()

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(25))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), game_training.train)
    winner = population.run(pe.evaluate, game_training.generations)



    # save_winner(winner)
    print(config)
