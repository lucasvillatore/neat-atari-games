import neat
import multiprocessing
import gym
import pickle
import argparse
from demon_attack.demon_attack import DemonAttack
from flappy_bird.flappy_bird import FlappyBird
from montezuma_revenge.montezuma_revenge import MontezumaRevenge
from pacman.pacman import Pacman
from pong.pong import Pong
from seaquest.seaquest import Seaquest

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

    def save_winner(self, winner):
        with open('winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)

class TrainingInterface:
    def __init__(self):
        self.GENERATIONS_DEFAULT = 100
        self.games = {
            "demon_attack": DemonAttack(),
            "flappy_bird": FlappyBird(),
            "montezuma_revenge": MontezumaRevenge(),
            "pacman": Pacman(),
            "pong": Pong(),
            "seaquest": Seaquest(),
        }

    def get_parameters(self):
        choices = self.games.keys()

        parser = argparse.ArgumentParser(description='Training to play space invaders')
        parser.add_argument('--generations', type=int, help='Number of generations', required=False, default=self.GENERATIONS_DEFAULT)
        parser.add_argument(
            '--game', 
            help='Game to training', 
            required=True, 
            choices=list(choices)
        )
        args = parser.parse_args()
        return args
    
    def get_game(self, game):
        try:
            return self.games[game] 
        except Exception as err:
            print(str(err))
            exit()


if __name__ == '__main__':
    training_interface = TrainingInterface()
    parameters = training_interface.get_parameters()
    
    game_instance = training_interface.get_game(parameters.game)
    game_training = Training(game=game_training, generations=parameters.generations)
    
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
