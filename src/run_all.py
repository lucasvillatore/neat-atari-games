
from trainer import run
# from pacman.interface import Pacman
# from alien.interface import Alien
# from space_invaders.interface import SpaceInvaders
# from assault.interface import Assault
from pong.interface import Pong
from breakout.interface import Breakout

if __name__ == '__main__':

    games = [

        # Pong(net='./pong/network/winner.pkl'),
        Breakout(net='./pong/network/basic-100gen.pkl'),
    ]

    for game in games:
        run(game)