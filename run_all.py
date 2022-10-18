
from trainer import run
import os 
if __name__ == '__main__':

    games = [
        {"game": "ALE/MsPacman-ram-v5", "config": "./pacman/configs/neat-config", 'folder': './pacman'},
        # {"game": "Pendulum-v1", "config": "./pendulum/configs/neat-config", 'folder': './pendulum'},
        {"game": "MountainCar-v0", "config": "./mountain_car/configs/neat-config", 'folder': './mountain_car'},
        {"game": "CartPole-v1", "config": "./cart_pole/configs/neat-config", 'folder': './cart_pole'},
        {"game": "Acrobot-v1", "config": "./acrobot/configs/neat-config", 'folder': './acrobot'},
        {"game": "SpaceInvaders-ram-v0", "config": "./space_invaders/configs/neat-config", 'folder': './space_invaders'},
    ]


    for game in games:
        run(game['game'], game['config'])
        os.rename('speciation.csv', game['folder']+'/speciation.csv')
        os.rename('species_fitness.csv', game['folder']+'/species_fitness.csv')
        os.rename('fitness_history.csv', game['folder']+'/fitness_history.csv')