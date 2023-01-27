
from trainer import run
from pong.interface import Pong
from breakout.interface import Breakout
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        Skiing(),
        # Breakout(net='./pong/network/winner.pkl'),
        # Pong(),
        # Tennis()
    ]

    for game in games:
        run(game)