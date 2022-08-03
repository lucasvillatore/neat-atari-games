import argparse
from pydoc import describe
from random import randrange
from time import sleep
from src.game import GameGym
from my_neat  import NeatTraining
import cv2
import neat
import gym
import json
import numpy as np

def train_genomes(episodes):
    game = 'SpaceInvaders-v4'
    env = gym.make(game)
    actions = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}
    
    for episode in range(1, episodes + 1):
        n_state = env.reset()
        
        done = False
        height, width, channels = n_state.shape 
        
        height = int(height * 50 / 100)
        width = int(width * 50 / 100)
        
        dim = (width, height)
        # neat_network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, configuration)
        while not done:
            img = cv2.cvtColor(n_state, cv2.COLOR_BGR2RGB)
            print("Oi")
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("img", resized)
            cv2.waitKey(0)
            exit()
        # while not done:
        #     img = cv2.cvtColor(n_state, cv2.COLOR_BGR2RGB)
        
        #     n_state, reward, done, info = env.step(randrange(6))
        #     print(env.observation_space)
        #     score += reward
        # fitness = score
        # print('Episode: {} Score: {}'.format(episode, score))
        
if __name__ == '__main__':

    train_genomes(100)