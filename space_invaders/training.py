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


GENERATIONS_DEFAULT = 100
GAME = 'ALE/SpaceInvaders-v5'
FRAMESKIP=3
INPUTS = 100
FRAME_ACTION = 3
kernel = np.ones((4,4),np.uint8)
actions = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}

frame = 0


if ('LOG_LEVEL' in os.environ and os.environ['LOG_LEVEL'] == 'debug') or \
    ('DEBUG' in os.environ and os.environ['DEBUG'] == 'true'):
    logging.basicConfig(
        format='%(levelname)s - %(asctime)s - %(message)s', 
        level=logging.DEBUG        
    )
    logging.addLevelName(logging.DEBUG, levelName='DEBUG')
else:
    logging.basicConfig(
        format='%(levelname)s - %(asctime)s - %(message)s', 
        level=logging.INFO
    )


class Mock:
    def activate(self, parameter):
        tmp = []
        for i in range(6):
            tmp.append(randrange(60))
        return tmp

def get_parameters():
    parser = argparse.ArgumentParser(description='Training to play space invaders')
    parser.add_argument('--generations', type=int, help='Number of generations', required=False, default=GENERATIONS_DEFAULT)
    args = parser.parse_args()
    
    return args

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
            coordinates.append([cX,cY])
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

    return monsters, shots, my_position 

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
        'coordinates': {},
        'actions': [],
        "lives": [
            {
                "frames_alive": 0,
                "quantity": 3,
                "multiplier": 0.05
            },
            {
                "frames_alive": 0,
                "quantity": 2,
                "multiplier": 0.03
            },
            {
                "frames_alive": 0,
                "quantity": 1,
                "multiplier": 0.01
            },
            {
                "frames_alive": 0,
                "quantity": 0,
                "multiplier": 0.00
            },
        ]
    }

def all_coordinates(coordinates):
    my_position = coordinates['my_position']
    shots = coordinates['shots']
    monsters = coordinates['monsters']
    
    all_coordinates = np.zeros(int(160/4))


    for monster in monsters:
        # print(my_position[0][0])
        # print(monster[0])
        # print(abs(my_position[0][0] - monster[0]))
        if abs(my_position[0][0] - monster[0]) <= 10:
            for index in range(-2, 3):
                if monster[0]/4 + index >= 0 and monster[0]/4 + index <= 160:
                    # all_coordinates[monster[0] + index] += 1
                    all_coordinates[int(monster[0]/4) + index] += 1
    # print(all_coordinates)
    # exit()
    
    # for shot in shots:
    #     #  for index in range(-1, 2):
    #     #     if shot[0] + index >= 0 and shot[0] + index <= 160:
    #     #         all_coordinates[shot[0] + index] += 10
    #     all_coordinates[int(shot[0]/4)] += 10

    # for position in my_position:
    #     # for index in range(-2, 4):
    #     #     if position[0] + index >= 0 and position[0] + index <= 160:
    #     #         all_coordinates[position[0] + index] += 30
    #     all_coordinates[int(position[0]/4)] += 100

    # print(all_coordinates)
    # exit(0)
    return all_coordinates


def run_game(environment, network):
    global frame
    n_state = environment.reset(seed=randrange(10000))

    game_information = init_game_information(n_state)
    
    number_of_lifes = 3
    while not game_information["done"]:
        for lives in range(4):
            if game_information['lives'][lives]['quantity'] == number_of_lifes:
                game_information['lives'][lives]['frames_alive'] += 1
        frame += 1
        if not another_action_is_avaliable(game_information):
            game_information["state"], game_information["reward"], game_information["done"], game_information["info"] = environment.step(action)
            game_information["score"] += game_information["reward"]

            number_of_lifes = game_information['info']['lives']
            continue
        
        monsters, shots, my_position = get_coordinates_from_image(game_information["state"])

        game_information['coordinates']['monsters'] = monsters
        game_information['coordinates']['shots'] = shots
        game_information['coordinates']['my_position'] = my_position

        ai_decision = network.activate(
            all_coordinates(game_information["coordinates"])
        )
        # logging.info(ai_decision)
        action = np.argmax(ai_decision)
        if action == 2:
            action = 4
        if action == 3:
            action = 5

        game_information["actions"].append(action)

        logging.debug("Action is {}".format(actions[action]))
        
        game_information["state"], game_information["reward"], game_information["done"], game_information["info"] = environment.step(action)
        game_information["score"] += game_information["reward"]
        number_of_lifes = game_information['info']['lives']

    environment.close()

    return game_information

def calculate_fitness(game_information):

    fitness = 0

    # fitness += game_information['info']['lives'] * 20
    # fitness += -game_information['info']['episode_frame_number']*0.01
    fitness += game_information['score']
    # fitness += 10000 if len(game_information['coordinates']['monsters']) == 0 else 0
    
    # for lives in range(4):
    #     fitness += game_information['lives'][lives]['frames_alive'] * game_information['lives'][lives]['multiplier']
    
    return fitness

def get_total_actions(game_actions):
    
    total_actions = {}
    for id, action in actions.items():
        total_actions[action] = game_actions.count(id)
    
    tmp = ""
    for action, total in total_actions.items():
        tmp += "{}: {} - ".format(action, total)
    
    return tmp

def train_genome(genome, configuration):
    env = gym.make(GAME, frameskip=FRAMESKIP)
    neat_network = neat.nn.feed_forward.FeedForwardNetwork.create(genome, configuration)

    game_information = run_game(env, neat_network)
    
    fitness = calculate_fitness(game_information)
    
    logging.info('Score: {} - Vidas Restantes: {} - Frames Vivo: {} - Monstros mortos: {} - actions {}'.format(
        fitness, 
        game_information["info"]["lives"],
        game_information['info']['episode_frame_number'],
        36 - len(game_information["coordinates"]["monsters"]),
        get_total_actions(game_information["actions"])
    ))

    return fitness

def save_winner(winner):
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

if __name__ == '__main__':
    if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'true':
        logging.debug("Using Mock network")
        winner_net = Mock()
        env = gym.make(GAME, frameskip=FRAMESKIP)
        run_game(env, winner_net)

    else:
        parameters = get_parameters()
        generations = parameters.generations

        neat_configuration = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                'configs/neat-config'
        )

        logging.info('Setting population')
        population = neat.Population(neat_configuration)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.Checkpointer(25))

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), train_genome)
        logging.info('Running train_genomes {} generations'.format(generations))
        winner = population.run(pe.evaluate, generations)
        save_winner(winner)
        
        winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_configuration)