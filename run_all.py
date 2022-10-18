
from trainer import run
import os 
if __name__ == '__main__':

    games = [
        {"game": "SpaceInvaders-ram-v0", "config": "./space_invaders/configs/neat-config", 'folder': './space_invaders'}
    ]


    for game in games:
        run(game['game'], game['config'])
        os.rename('speciation.csv', game['folder']+'/speciation.csv')
        os.rename('species_fitness.csv', game['folder']+'/species_fitness.csv')
        os.rename('fitness_history.csv', game['folder']+'/fitness_history.csv')