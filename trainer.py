from neat import nn, population, parallel
from dotenv import load_dotenv
import numpy as np
import gym
import os

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
    fitnesses = []

    for runs in range(episodes):
        inputs = environment.reset()
        total_reward = 0.0
        
        for j in range(steps):
            outputs = net.serial_activate(inputs)
            action = np.argmax(outputs)
            inputs, reward, done, info = env.step(action)
            total_reward += game.calculate_fitness(reward)

            if done:
                break
            
        fitnesses.append(total_reward)

    fitness = np.array(fitnesses).mean()
    
    print("Species fitness: %s" % str(fitness))
    
    return fitness

def evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    
    return simulate_species(net, environment, config.episodes, config.max_steps)

def eval_fitness(genomes):
    for g in genomes:
        fitness = evaluate_genome(g)
        g.fitness = fitness

def setup_population():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config.path)
    pop = population.Population(config_path)

    if game.checkpoint is not None:
        print("Checkpoint loaded")
        pop.load_checkpoint(game.checkpoint)

    return pop

def run_trainer(trainer_population):
    global environment

    if config.render:
        environment = gym.make(config.game, render_mode="human", full_action_space=True)
        trainer_population.run(eval_fitness, config.generations)
        return

    parallel_train = parallel.ParallelEvaluator(config.num_cores, evaluate_genome)
    trainer_population.run(parallel_train.evaluate, config.generations)

def train_network():
    trainer_population = setup_population()

    run_trainer(trainer_population)
    
    game.save_checkpoint(trainer_population)
    game.save_statistics(trainer_population)
    game.save_winner(trainer_population)
    
    print('Number of evaluations: {0}'.format(trainer_population.total_evaluations))

def replay(env, winner):
    winner_net = nn.create_feed_forward_phenotype(winner)
    env = gym.make(config.game, render_mode="human")
    
    simulate_species(winner_net, env, 1, config.max_steps)

def run(game_instance):
    global game, environment, config
    config = Config(game=game_instance.name, config=game_instance.config)
    environment = gym.make(config.game, full_action_space=True)
    game = game_instance
    
    train_network()