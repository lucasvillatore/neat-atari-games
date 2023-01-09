
from trainer import run
# from pacman.interface import Pacman
# from alien.interface import Alien
# from space_invaders.interface import SpaceInvaders
# from assault.interface import Assault
from pong.interface import Pong
from breakout.interface import Breakout

if __name__ == '__main__':

    games = [

        # Breakout(name="Breakout-v4", neat_config_path="./breakout/configs/neat-config", folder="./breakout"),
        Pong(name='Pong-v4', neat_config_path='./pong/configs/neat-config', folder='./pong')
        # Pacman(name="ALE/MsPacman-ram-v5", config= "./pacman/configs/neat-config",folder='./pacman') ,
        # Alien(name="ALE/Alien-ram-v5", config="./alien/configs/neat-config", folder='./alien'),

        
        # Pacman(name="ALE/MsPacman-ram-v5", config= "./pacman/configs/neat-config",folder='./pacman', checkpoint='./alien/checkpoint') ,
        # Alien(name="ALE/Alien-ram-v5", config="./alien/configs/neat-config", folder='./alien', checkpoint='./pacman/checkpoint'),

        # SpaceInvaders(name="ALE/SpaceInvaders-ram-v5", config="./space_invaders/configs/neat-config", folder='./space_invaders'),
        # Assault(name="ALE/Assault-ram-v5", config="./assault/configs/neat-config", folder='./assault'),


        # MountainCar(name="MountainCar-v0", config= "./mountain_car/configs/neat-config", folder='./mountain_car'),
        # CartPole(name="CartPole-v0", config= "./cart_pole/configs/neat-config", folder= './cart_pole'),
        # Acrobot(name="Acrobot-v1", config= "./acrobot/configs/neat-config", folder= './acrobot'),
    ]

    for game in games:
        run(game)