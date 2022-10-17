import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel
from dotenv import load_dotenv
import pickle

load_dotenv()

class Config():
    def __init__(self):
        self.game = os.environ['GAME']
        self.generations = int(os.environ['GENERATIONS'])
        self.max_steps = int(os.environ['MAX_STEPS'])
        self.episodes = int(os.environ['EPISODES'])
        self.render = bool(int(os.environ['RENDER']))
        self.num_cores = int(os.environ['NUM_CORES'])
        self.checkpoint = bool(int(os.environ['CHECKPOINT']))

def simulate_species(net, env, episodes=1, steps=5000):
    fitnesses = []
    for runs in range(episodes):
        inputs = my_env.reset()
        cum_reward = 0.0
        for j in range(steps):

            outputs = net.serial_activate(inputs)
            action = np.argmax(outputs)
            inputs, reward, done, _ = env.step(action)
            if done:
                break
            cum_reward += reward
        print(cum_reward)
        fitnesses.append(cum_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness


def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, my_env, config.episodes, config.max_steps)

def evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, my_env, config.episodes, config.max_steps)

def eval_fitness(genomes):
    for g in genomes:
        fitness = evaluate_genome(g)
        g.fitness = fitness

def save_winner(winner):
    with open('winner.pkl', 'wb') as output:
       pickle.dump(winner, output, 1)

def replay(env, winner):
    winner_net = nn.create_feed_forward_phenotype(winner)
    env = gym.make(config.game, render_mode="human")
    for i in range(100):
        simulate_species(winner_net, env, 1, config.max_steps)

def train_network(env):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'gym_config')
    pop = population.Population(config_path)

    if config.checkpoint:
        pop.load_checkpoint('checkpoint')

    if config.render:
        env = gym.make(config.game, render_mode="human")
        pop.run(eval_fitness, config.generations)
    else:
        pe = parallel.ParallelEvaluator(config.num_cores, worker_evaluate_genome)
        pop.run(pe.evaluate, config.generations)

    pop.save_checkpoint("checkpoint")

    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    winner = pop.statistics.best_genome()

    save_winner(winner)

    print('\nBest genome:\n{!s}'.format(winner))
    replay(env, winner)


config = Config()

my_env = gym.make(config.game)

if __name__ == '__main__':

    train_network(my_env)