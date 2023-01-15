import numpy as np
import neat
import gym
import os
from dotenv import load_dotenv
from common import visualize
from common.atariari.benchmark.wrapper import AtariARIWrapper
import pickle

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
        environment = AtariARIWrapper(gym.make(trainer_config.game, render_mode="human", obs_type="ram"))
    else:
        environment = AtariARIWrapper(gym.make(trainer_config.game, obs_type="ram"))

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

    pop = neat.Population(neat_configuration)

    # if game.checkpoint is not None:
    #     print("Checkpoint loaded")
    #     pop.load_checkpoint(game.checkpoint)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = run_trainer(pop)

    visualize.draw_net(neat_configuration, winner)
    visualize.plot_stats(stats, ylog=False)
    visualize.plot_species(stats)

    with open('winner.pkl', 'wb') as output:
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
    net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config_neat)
    
    return simulate_species(net, trainer_config.episodes, trainer_config.max_steps)

def simulate_species(net, episodes=1, steps=5000):
    fitnesses = []
    
    for runs in range(episodes):
        total_reward = game.run(net, environment, steps)
        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    
    return fitness




if __name__ == '__main__':
    from pong.interface import Pong

    game_instance = Pong(name='Pong-v4', neat_config_path='./pong/configs/neat-config', folder='./pong')

    trainer_config = TrainerConfig(game=game_instance.name, neat_config_path=game_instance.neat_config_path)

    if trainer_config.render:
        environment = AtariARIWrapper(gym.make(trainer_config.game, render_mode="human", obs_type="ram"))
    else:
        environment = AtariARIWrapper(gym.make(trainer_config.game, obs_type="ram"))

    game = game_instance
    
    with open("winner.pkl", "rb") as f:
        winner = pickle.load(f)
    
    neat_configuration = neat.Config(
            neat.DefaultGenome, 
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation,
            "./pong/configs/neat-config"
    )


    pop = neat.Population(neat_configuration)

    tmp = worker_genome(winner, neat_configuration)




    
# def replay(env, winner):
#     winner_net = neat.nn.recurrent.RecurrentNetwork.create(winner)
#     env = gym.make(config.game, render_mode="human", obs_type="ram")
    
#     simulate_species(winner_net, env, 1, config.max_steps)