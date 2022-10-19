
from trainer import run
import os 


if __name__ == '__main__':

    games = [
        # {"game": "ALE/MsPacman-ram-v5", "config": "./pacman/configs/neat-config", 'folder': './pacman'},
        # {"game": "ALE/Alien-ram-v5", "config": "./alien/configs/neat-config", 'folder': './alien'},
        # {"game": "ALE/SpaceInvaders-ram-v5", "config": "./space_invaders/configs/neat-config", 'folder': './space_invaders'},
        # {"game": "ALE/Assault-ram-v5", "config": "./assault/configs/neat-config", 'folder': './assault'},


        # {"game": "Pendulum-v1", "config": "./pendulum/configs/neat-config", 'folder': './pendulum'},
        # {"game": "MountainCar-v0", "config": "./mountain_car/configs/neat-config", 'folder': './mountain_car'},
        # {"game": "CartPole-v0", "config": "./cart_pole/configs/neat-config", 'folder': './cart_pole'},
        # {"game": "Acrobot-v1", "config": "./acrobot/configs/neat-config", 'folder': './acrobot'},
    ]


    for game in games:
        run(game['game'], game['config'])
        os.rename('speciation.csv', game['folder']+'/speciation.csv')
        os.rename('species_fitness.csv', game['folder']+'/species_fitness.csv')
        os.rename('fitness_history.csv', game['folder']+'/fitness_history.csv')
        os.rename('checkpoint', game['folder'] + '/checkpoint')
        os.rename('winner.pkl', game['folder'] + '/winner.pkl')