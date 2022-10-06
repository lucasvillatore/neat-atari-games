import neat
import multiprocessing
import gym
import pickle
import argparse
import os
import logger
import logging
from dotenv import load_dotenv
from demon_attack.demon_attack import DemonAttack
from flappy_bird.flappy_bird import FlappyBird
from montezuma_revenge.montezuma_revenge import MontezumaRevenge
from pacman.pacman import Pacman
from pong.pong import Pong
from seaquest.seaquest import Seaquest


load_dotenv()
logger.load()

GENERATIONS_DEFAULT = 100

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
        self.games = {
            "demon_attack": DemonAttack(),
            "flappy_bird": FlappyBird(),
            "montezuma_revenge": MontezumaRevenge(),
            "pacman": Pacman(),
            "pong": Pong(),
            "seaquest": Seaquest(),
        }
        if 'GAME' in os.environ:
            try:
                self.game = self.games[os.environ['GAME']]
            except Exception as err:
                logging.error(str(err))
                exit()
        else:
            logging.error("environment variable GAME is not setted")
            exit()

        self.generations = GENERATIONS_DEFAULT
        if 'GENERATIONS' in os.environ:
            self.generations = int(os.environ['GENERATIONS'])
    
    
    def get_game(self):
        return self.game


if __name__ == '__main__':
    training_interface = TrainingInterface()    
    game_training = Training(
        game=training_interface.get_game(), 
        generations=training_interface.generations
    )
    
    config = game_training.game.get_neat_configuration()

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(25))


    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), game_training.train)
    winner = population.run(pe.evaluate, game_training.generations)


    game_training.save_winner(winner)
    print(config)
