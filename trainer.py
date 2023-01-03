import numpy as np
import neat
import gym
import os
from dotenv import load_dotenv
from common import visualize
from atariari.benchmark.wrapper import AtariARIWrapper

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

def convert(o):
    if isinstance(o, np.uint8): 
        return int(o)  
    raise TypeError

def simulate_species(net, env, episodes=1, steps=5000):
    global environment
    fitnesses = []
    
    import json
    actions = {0: 0, 1: 0, 2: 0, 3: 0}
    for runs in range(episodes):
        game_environment = environment.reset()

        total_reward = 0.0
        step = 0
        info = {}
        for j in range(steps):
            
            action = game.run_step(game_environment, net, env, step, info)
            actions[action] += 1
            game_environment, reward, done, info = env.step(action)
            total_reward += game.calculate_fitness(reward)

            step += 1
            if done:
                break
            
        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    
    print(actions)
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
    visualize.draw_net(config_neat, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    
def replay(env, winner):
    winner_net = neat.nn.create_feed_forward_phenotype(winner)
    env = gym.make(config.game, render_mode="human")
    
    simulate_species(winner_net, env, 1, config.max_steps)

def run(game_instance):
    global game, environment, config
    config = Config(game=game_instance.name, config=game_instance.config)

    if config.render:
        environment = AtariARIWrapper(gym.make(config.game, render_mode="human", obs_type="ram"))
    else:
        environment = AtariARIWrapper(gym.make(config.game, obs_type="ram"))

    game = game_instance
    
    train_network()