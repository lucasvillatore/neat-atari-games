
from trainer import run
from pong.interface import Pong
from breakout.interface import Breakout;
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        # Pong(folder='./pong/pong_y_full_action_space'),
        Breakout(folder='./breakout/breakout_pong_y_full_action_space', checkpoint='./pong/pong_y_full_action_space/checkpoints/neat-checkpoint-29'),
        # Skiing(),
        # Tennis(
        #     folder="./tennis/tenis_x",
            # checkpoint='./tennis/tenis_x/checkpoints/neat-checkpoint-29'
        # ),
        # Tennis(
        #     folder="./tennis/tenis_pong_y_full_action_space",
        #     checkpoint='./pong/pong_y_full_action_space/checkpoints/neat-checkpoint-29'
        # ),
        # Tennis(
        #     folder="./tennis/tenis_pong_y",
        #     checkpoint='./pong/pong_y/checkpoints/neat-checkpoint-29'
        # ),
    ]

    for game in games:
        run(game)