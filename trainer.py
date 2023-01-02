from neat import nn, population, parallel
from dotenv import load_dotenv
import numpy as np
import gym
import os
import cv2 as cv
import numpy as np

load_dotenv()

config = None
environment = None
game = None

kernel = np.ones((2,2), np.uint8) 
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
            gray_image = cv.cvtColor(game_environment, cv.COLOR_BGR2GRAY)

            _, bw_img = cv.threshold(gray_image, 40, 255, cv.THRESH_BINARY)

            game_dilatado = cv.dilate(bw_img, kernel, iterations=1)
            teste = game_dilatado.copy()


            try:
                outputs = net.serial_activate(teste)
                action = np.argmax(outputs)
            except Exception as err:
                action = 0
            game_environment, reward, done, info = env.step(action)
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

    if config.render:
        environment = gym.make(config.game, render_mode="human")
    else:
        environment = gym.make(config.game)

    game = game_instance
    
    train_network()