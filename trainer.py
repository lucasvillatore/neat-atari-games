import numpy as np
import neat
import gym
import os
from dotenv import load_dotenv
from common import visualize

load_dotenv()

config = None
environment = None
game = None

class Config():
    def __init__(self, game, config):
        self.game = game
        self.path = config
        self.generations = int(os.environ['GENERATIONS'])
        self.max_steps = int(os.environ['MAX_STEPS'])
        self.episodes = int(os.environ['EPISODES'])
        self.render = bool(int(os.environ['RENDER']))
        self.num_cores = int(os.environ['NUM_CORES'])

def simulate_species(net, env, episodes=1, steps=5000):
    global environment
    fitnesses = []

    for runs in range(episodes):
        game_environment = environment.reset()

        total_reward = 0.0
        for j in range(steps):
            game_environment, reward, done, info = game.run_step(game_environment, net, env)
            total_reward += game.calculate_fitness(reward)

            if done:
                break
            
        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    
    print("Species fitness: %s" % str(fitness))
    
    return fitness

def worker_genome(genome, config_neat):
    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config_neat)
    
    return simulate_species(net, environment, config.episodes, config.max_steps)

def eval_fitness(genomes, config):
    for genome_id, genome in genomes:
        fitness = worker_genome(genome, config)
        genome.fitness = fitness


def running_in_multiples_cores():
    return int(config.num_cores) == 1

def run_trainer(trainer_population):
    if running_in_multiples_cores:
        return trainer_population.run(eval_fitness, config.generations)

    parallel_train = neat.parallel.ParallelEvaluator(config.num_cores, worker_genome)
    return trainer_population.run(parallel_train.evaluate, config.generations)

def train_network():
    config_neat = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config.path)

    pop = neat.Population(config_neat)

    if game.checkpoint is not None:
        print("Checkpoint loaded")
        pop.load_checkpoint(game.checkpoint)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = run_trainer(pop)

    node_names = {0: "noop", 1: "fire", 2: "right", 3: "left"}
    visualize.draw_net(config_neat, winner, False, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)
    
def replay(env, winner):
    winner_net = neat.nn.create_feed_forward_phenotype(winner)
    env = gym.make(config.game, render_mode="human")
    
    simulate_species(winner_net, env, 1, config.max_steps)

def run(game_instance):
    global game, environment, config
    config = Config(game=game_instance.name, config=game_instance.config)

    if config.render:
        environment = gym.make(config.game, render_mode="human")
    else:
        environment = gym.make(config.game)

    game = game_instance
    
    train_network()