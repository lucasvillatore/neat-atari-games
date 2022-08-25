import os
import gym
import cv2
import neat
import pickle
import logging
import imutils
import argparse
import numpy as np
import multiprocessing
from random import randrange

GAME = 'ALE/SpaceInvaders-v5'
INPUTS = 100
FRAME_ACTION = 5
kernel = np.ones((4,4),np.uint8)

actions = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}

frame = 0

def get_coordinates(state):
    dilation = cv2.dilate(state,kernel,iterations = 1)
    blurred = cv2.GaussianBlur(dilation, (5, 5), 0)
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

    return coordinates

def get_monsters_coordinates(state):
    monster_rgb = np.array([134, 134, 29])
    monster_state = cv2.inRange(state, monster_rgb, monster_rgb)

    monster_coordinates = get_coordinates(monster_state)

    return monster_coordinates

def get_shots_coordinates(state):
    shot_rgb = np.array([142,142,142])
    shots_state = cv2.inRange(state, shot_rgb, shot_rgb)

    shots_coordinates = get_coordinates(shots_state)

    return shots_coordinates

def get_my_position_coordinates(state):
    my_position_rgb = np.array([50, 132, 50])
    my_position_state = cv2.inRange(state, my_position_rgb, my_position_rgb)

    my_position_coordinates = get_coordinates(my_position_state)

    return my_position_coordinates


def get_coordinates_from_image(state):
    monsters = get_monsters_coordinates(state)
    shots = get_shots_coordinates(state)
    my_position = get_my_position_coordinates(state)
    
    logging.debug("Monsters coordinates {}".format(monsters))
    logging.debug("Shots coordinates {}".format(shots))
    logging.debug("My position coordinates {}".format(my_position))

    coordinates = my_position + shots + monsters 
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
        'coordinates': None,
    }

def run_game(environment, network):
    global frame
    n_state = environment.reset(seed=randrange(10000))
    game_information = init_game_information(n_state)
    
    while not game_information["done"]:
        frame += 1
        if not another_action_is_avaliable(game_information):
            game_information["state"], game_information["reward"], game_information["done"], game_information["info"] = environment.step(action)
            game_information["score"] += game_information["reward"]
            continue
        
        game_information["coordinates"] = get_coordinates_from_image(game_information["state"])
        ai_decision = network.activate(game_information["coordinates"])
        action = np.argmax(ai_decision)

        logging.debug("Action is {}".format(actions[action]))
        
        game_information["state"], game_information["reward"], game_information["done"], game_information["info"] = environment.step(action)
        game_information["score"] += game_information["reward"]

    environment.close()

    return game_information
    
neat_configuration = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'configs/neat-config'
)
with open("winner.pkl", "rb") as f:
    winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_configuration)

    env = gym.make(GAME, render_mode="human", frameskip=1)
    run_game(env, winner_net)