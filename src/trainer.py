import numpy as np
import neat
import gym
import os
from dotenv import load_dotenv
from common import visualize
from common.atariari.benchmark.wrapper import AtariARIWrapper
import pickle
from pong.interface import Pong
from breakout.interface import Breakout
from tennis.interface import Tennis
from skiing.interface import Skiing




load_dotenv()

trainer_config = None
environment = None
game = None

class TrainerConfig():
    def __init__(self, game, neat_config_path):
        self.game = game
        self.neat_config_path = neat_config_path
        self.generations = int(os.environ['GENERATIONS'])
        self.max_steps = int(os.environ['MAX_STEPS'])
        self.episodes = int(os.environ['EPISODES'])
        self.render = bool(int(os.environ['RENDER']))
        self.num_cores = int(os.environ['NUM_CORES'])

def run(game_instance):
    global game, environment, trainer_config
    trainer_config = TrainerConfig(game=game_instance.name, neat_config_path=game_instance.neat_config_path)

    if trainer_config.render:
        environment = AtariARIWrapper(gym.make(trainer_config.game, render_mode="human", obs_type="ram", full_action_space=True))
    else:
        environment = AtariARIWrapper(gym.make(trainer_config.game, obs_type="ram", full_action_space=True))

    game = game_instance
    
    train_network()

def train_network():
    neat_configuration = neat.Config(
            neat.DefaultGenome, 
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation,
            trainer_config.neat_config_path
    )

    if game.checkpoint is None:
        pop = neat.Population(neat_configuration)
    else:
        pop = neat.Checkpointer.restore_checkpoint(game.checkpoint)
        print(f"Checkpoint {game.checkpoint} loaded")

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(100, filename_prefix=f"{game.folder}/checkpoints/neat-checkpoint-"))

    winner = run_trainer(pop)

    stats.save_genome_fitness(filename=f"{game.folder}/statistics/fitness_history.csv")
    visualize.draw_net(neat_configuration, winner, filename=f"{game.folder}/graphs/{game.name}-net")
    visualize.plot_stats(stats, ylog=False, filename=f"{game.folder}/graphs/{game.name}-avg_fitness.svg")
    visualize.plot_species(stats, filename=f"{game.folder}/graphs/{game.name}-speciation.svg")

    with open(f'{game.folder}/network/winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

def run_trainer(trainer_population):
    running_in_multiples_cores = int(trainer_config.num_cores) == 1

    if running_in_multiples_cores:
        return trainer_population.run(eval_fitness, trainer_config.generations)

    parallel_train = neat.parallel.ParallelEvaluator(trainer_config.num_cores, worker_genome)
    return trainer_population.run(parallel_train.evaluate, trainer_config.generations)

def eval_fitness(genomes, config):
    for genome_id, genome in genomes:
        fitness = worker_genome(genome, config)
        genome.fitness = fitness

def worker_genome(genome, config_neat):
    if game.net is None:
        net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config_neat)
    else:
        print(f"Loading {game.net}")
        with open(game.net, 'rb') as f:
            pickle_net = pickle.load(f)
        net = neat.nn.feed_forward.FeedForwardNetwork.create(pickle_net, config_neat)

    return simulate_species(net, trainer_config.episodes, trainer_config.max_steps)

def simulate_species(net, episodes=1, steps=5000):
    fitnesses = []
    
    for runs in range(episodes):
        total_reward = game.run(net, environment, steps)
        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    
    return fitness


def get_game(name):

    games_dict = {
        'breakout': Breakout(),
        'pong': Pong(),
        'tennis': Tennis(),
        'skiing': Skiing()
    }

    return games_dict[name] 

if __name__ == '__main__':
    game_instance = get_game(os.environ['GAME'])

    run(game_instance)