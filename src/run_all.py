
from trainer import run
from pong.interface import Pong
from breakout.interface import Breakout
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        # Breakout(checkpoint='./pong/checkpoints/pong-x-y/neat-checkpoint-293'),
        # Pong(),
        # Skiing(),
        Tennis(net='./tennis/network/winner.pkl'),
    ]

    for game in games:
        run(game)