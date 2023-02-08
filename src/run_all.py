
from trainer import run
from pong.interface import Pong
from breakout.interface import Breakout;
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        # Pong(folder='./pong/pong_x_y'),
        # Breakout(
        #     folder='./breakout/breakout_x_y', 
        # ),
        Tennis(
            folder="./tennis/tennis_x_y",
        ),
        # Skiing(
        #     folder='./skiing/skiing_x',
        #     checkpoint='./pong/pong_y/checkpoints/neat-checkpoint-29'
        # )
    ]

    for game in games:
        run(game)