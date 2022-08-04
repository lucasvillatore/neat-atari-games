import argparse
import os
from random import randrange
from time import sleep
from src.game import GameGym
from my_neat  import NeatTraining
import cv2
import neat
import gym
import json
import numpy as np
from datetime import datetime

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
        logger("Running episode {}".format(episode), True)
        n_state = env.reset()
        done = False
        score = 0
        
        height, width, channels = n_state.shape 
        
        height = int(height * 50 / 100)
        width = int(width * 50 / 100)
        
        dim = (width, height)
        neat_network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, configuration)
        while not done:
            logger('Changing to gray')
            img = cv2.cvtColor(n_state, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            logger('After change')
            
            imgarray = np.ndarray.flatten(resized)
            
            logger('Activating network')
            ai_decision = neat_network.activate(imgarray)
            logger('After activate')
            
            action = np.argmax(ai_decision)
            logger(f"Action: {actions[action]}")
            
            
            n_state, reward, done, info = env.step(action)
            score += reward
        fitness = score
        genome.fitness = fitness
        
        logger('Episode: {} Score: {}'.format(episode, score), True)

def logger(message, show = False):
    if os.environ.get('debug', None) or show:
        current_time = datetime.now()
        print("{} - {}".format(current_time, message))
    
if __name__ == '__main__':
    # args = get_parameters()
    
    neat_configuration = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'configs/neat-config'
    )
    current_time = datetime.now()
    logger('Setting population', True)
    population = neat.Population(neat_configuration)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))
    current_time = datetime.now()
    logger('Running train_genomes', True)
    winner = population.run(train_genomes, 300)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_configuration)
    
    