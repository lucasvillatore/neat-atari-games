
from trainer import run
from pong.interface import Pong
from breakout.interface import Breakout
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        Breakout(net='./breakout/network/winner.pkl'),
        Pong(),
        # Skiing(),
        # Tennis(),
    ]

    for game in games:
        run(game)