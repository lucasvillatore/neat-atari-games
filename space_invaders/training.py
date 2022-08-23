import os
import gym
import cv2
import neat
import imutils
import argparse
import numpy as np
import multiprocessing
from datetime import datetime


GAME = 'ALE/SpaceInvaders-v5'
INPUTS = 100
FRAME_ACTION = 5

actions = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}


def get_parameters():
    parser = argparse.ArgumentParser(description='Training to play space invaders')
    parser.add_argument('--generations', type=int, help='Number of generations', required=True)
    args = parser.parse_args()
    
    return args

def get_coordinates_from_image(state):
    img = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    coordinates = []
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append(cX)
            coordinates.append(cY)
        except Exception as err:
            pass


    if len(coordinates) < INPUTS:
        for i in range(0, INPUTS - len(coordinates)):
            coordinates.append(0)

    return coordinates

def is_first_action(game_information):
    return game_information["action"] is None or game_information["info"] is None

def is_frame_action(game_information):
    return  game_information["info"]["frame_number"] % FRAME_ACTION == 0

def another_action_is_avaliable(game_information):
    if is_first_action(game_information) or is_frame_action(game_information):
        return True
    return False

def init_game_information(n_state):
    return {
        "score": 0, 
        "reward": 0,
        "action": None, 
        "state": n_state, 
        "done": False, 
        'info': None, 
    }

def run_game(environment, network):
    n_state = environment.reset()
    game_information = init_game_information(n_state)
    
    while not game_information["done"]:
        if not another_action_is_avaliable(game_information):
            game_information["state"], game_information["reward"], game_information["done"], game_information["info"] = environment.step(action)
            game_information["score"] += game_information["reward"]
            continue
        
        coordinates = get_coordinates_from_image(game_information["state"])
        ai_decision = network.activate(coordinates)
        action = np.argmax(ai_decision)
        game_information["state"], game_information["reward"], game_information["done"], game_information["info"] = environment.step(action)
        game_information["score"] += game_information["reward"]

    environment.close()

    return game_information

def train_genome(genome, configuration):
    env = gym.make(GAME)
    neat_network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, configuration)

    game_information = run_game(env, neat_network)
    
    fitness = game_information['info']['lives'] * 20 + game_information['info']['episode_frame_number']*0.01

    logger('Score: {}'.format(fitness), True)

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

    run_game(env, winner_net)