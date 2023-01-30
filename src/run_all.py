
from trainer import run
from pong.interface import Pong
from breakout.interface import Breakout
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        Skiing(),
        Breakout(),
        Tennis(),
        Pong(),
    ]

    for game in games:
        run(game)