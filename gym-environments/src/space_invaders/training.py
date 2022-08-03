import argparse
from random import randrange
from time import sleep
from src.game import GameGym
from my_neat  import NeatTraining
import cv2
import neat
import gym
import json
import numpy as np

def get_parameters():
    parser = argparse.ArgumentParser(description='Training to play space invaders')
    parser.add_argument('--generations', type=int, help='Number of generations', required=True)
    args = parser.parse_args()
    
    return args

def train_genomes(episodes, configuration):
    game = 'SpaceInvaders-v0'
    env = gym.make(game)
    actions = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}
    
    for episode, genome in episodes:
        n_state = env.reset()
        done = False
        score = 0
        
        height, width, channels = n_state.shape 
        
        height = int(height * 50 / 100)
        width = int(width * 50 / 100)
        
        dim = (width, height)
        neat_network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, configuration)
        while not done:
            img = cv2.cvtColor(n_state, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            imgarray = np.ndarray.flatten(resized)
            
            ai_decision = neat_network.activate(imgarray)
            action = np.argmax(ai_decision)
            print(f"Action: {actions[action]}")
            n_state, reward, done, info = env.step(action)
            score += reward
        fitness = score
        genome.fitness = fitness
        print('Episode: {} Score: {}'.format(episode, score))
    
if __name__ == '__main__':
    args = get_parameters()
    
    neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'configs/neat-config'
    )
    population = neat.Population(neat_configuration)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))
    winner = population.run(train_genomes, 1)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_configuration)
    
    