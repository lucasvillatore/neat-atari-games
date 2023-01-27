
from trainer import run
# from pacman.interface import Pacman
# from alien.interface import Alien
# from space_invaders.interface import SpaceInvaders
# from assault.interface import Assault
from pong.interface import Pong
from breakout.interface import Breakout
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        Skiing(net='./pong/network/winner.pkl'),
        # Breakout(net='./pong/network/winner.pkl'),
        # Pong(),
        # Tennis()
    ]

    for game in games:
        run(game)