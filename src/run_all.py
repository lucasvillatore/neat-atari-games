
from trainer import run
from pong.interface import Pong, PongXY, PongY
from tennis.interface import Tennis, TennisXY, TennisX

if __name__ == '__main__':

    games = [
        Tennis(
            folder="./tennis/tennis_pong_x_y_full_action_space",
            full_action_space=True,
            tmp=TennisX(),
            net="./tennis/tennis_pong_x_y_full_action_space/network/winner.pkl"
        ),
        # Tennis(
        #     folder="./tennis/tennis_pong_y_full_action_space",
        #     full_action_space=True,
        #     tmp=TennisX(),
        #     checkpoint="./pong/pong_y_full_action_space/checkpoints/neat-checkpoint-30"
        # ),
        # Tennis(
        #     folder="./tennis/tennis_pong_x_y",
        #     full_action_space=False,
        #     tmp=TennisXY(),
        #     checkpoint="./pong/pong_x_y/checkpoints/neat-checkpoint-30"
        # ),
        # Tennis(
        #     folder="./tennis/tennis_pong_x_y_full_action_space",
        #     full_action_space=True,
        #     tmp=TennisXY(),
        #     checkpoint="./pong/pong_x_y_full_action_space/checkpoints/neat-checkpoint-30"
        # ),
    ]

    for game in games:
        run(game)


