import argparse
import os
import multiprocessing
from random import randrange
import cv2
import neat
import gym
import numpy as np
from datetime import datetime
import imutils


GAME = 'ALE/SpaceInvaders-v5'
actions = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}
kernel = np.ones((5,5),np.uint8)
INPUTS = 100
def get_parameters():
    parser = argparse.ArgumentParser(description='Training to play space invaders')
    parser.add_argument('--generations', type=int, help='Number of generations', required=True)
    args = parser.parse_args()
    
    return args

def train_genome(genome, configuration):
    env = gym.make(GAME)
    n_state = env.reset()
    done = False
    score = 0
    
    height, width, channels = n_state.shape 
    
    height = int(height * 50 / 100)
    width = int(width * 50 / 100)
    
    dim = (width, height)
    neat_network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, configuration)
    info = {"frame_number": 0}
    action = None
    while not done:
        if info["frame_number"] % 3 == 0 and action is not None:
            n_state, reward, done, info = env.step(action)
            score += reward
            continue

        logger('Changing to gray')
        img = cv2.cvtColor(n_state, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        all_tmp = []
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                all_tmp.append(cX)
                all_tmp.append(cY)
            except Exception as err:
                print(M)
                print(c)


        if len(all_tmp) < INPUTS:
            for i in range(0, INPUTS-len(all_tmp)):
                all_tmp.append(0)
        
        logger('Activating network')
        ai_decision = neat_network.activate(all_tmp)
        logger('After activate')
        
        action = np.argmax(ai_decision)
        logger(f"Action: {actions[action]}")
        
        
        n_state, reward, done, info = env.step(action)
        score += reward
    fitness =info['lives'] * 20 + info['episode_frame_number']*0.001

    logger('Score: {}'.format(fitness), True)
    env.close()
    return fitness

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
    population.add_reporter(neat.Checkpointer(25))
    current_time = datetime.now()
    logger('Running train_genomes', True)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), train_genome)
    winner = population.run(pe.evaluate, 200)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_configuration)

    env = gym.make(GAME, render_mode='human')

    n_state = env.reset()
    done = False

    score = 0

    info = {"frame_number": 0}
    action = None
    while not done:

        if info["frame_number"] % 3 == 0 and action is not None:
            n_state, reward, done, info = env.step(action)
            score += reward
            continue

        logger('Changing to gray')
        img = cv2.cvtColor(n_state, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        all_tmp = []
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                all_tmp.append(cX)
                all_tmp.append(cY)
            except Exception as err:
                print(M)


        if len(all_tmp) < INPUTS:
            for i in range(0, INPUTS-len(all_tmp)):
                all_tmp.append(0)
            
            
        
        logger('Activating network')
        # ai_decision = winner_net.activate(imgarray)
        logger('After activate')
        
        # action = np.argmax(ai_decision)
        action = randrange(6)
        logger(f"Action: {actions[action]}", True)
        
        
        n_state, reward, done, info = env.step(action)
        score += reward