
from trainer import run
from pong.interface import Pong, PongXY, PongY
from breakout.interface import Breakout, BreakoutX, BreakoutXY, BreakoutPongXY, BreakoutPongY
from tennis.interface import Tennis
from skiing.interface import Skiing

if __name__ == '__main__':

    games = [
        Pong(
            folder='./pong/pong_testes_neat', 
            full_action_space=True,
            tmp=PongY()
        ),
        # Pong(
        #     folder='./pong/pong_x_y', 
        #     full_action_space=False,
        #     tmp=PongXY()
        # ),
        # Pong(
        #     folder='./pong/pong_y_full_action_space', 
        #     full_action_space=True,
        #     tmp=PongY()
        # ),
        # Breakout(
        #     folder='./breakout/breakout_x_y', 
        #     full_action_space=False,
        #     tmp=BreakoutXY()
        # ),
        # Breakout(
        #     folder='./breakout/breakout_pong_x_y', 
        #     full_action_space=False,
        #     tmp=BreakoutPongXY(),
        #     checkpoint='./pong/pong_x_y/checkpoints/neat-checkpoint-499'
        # ),
        # Breakout(
        #     folder='./breakout/breakout_x', 
        #     full_action_space=False,
        #     tmp=BreakoutX()
        # ),
        # Breakout(
        #     folder='./breakout/breakout_pong_y', 
        #     full_action_space=False,
        #     tmp=BreakoutPongY(),
        #     checkpoint='./pong/pong_y/checkpoints/neat-checkpoint-499'
        # ),
        # Breakout(
        #     folder='./breakout/breakout_pong_y_full_action_space', 
        #     full_action_space=True,
        #     tmp=BreakoutPongY(),
        #     checkpoint='./pong/pong_y_full_action_space/checkpoints/neat-checkpoint-445'
        # ),
        # Tennis(
        #     folder="./tennis/tennis_x_y",
        #     full_action_space=False
        # ),
        # Tennis(
        #     folder="./tennis/tennis_pong_x_y",
        #     full_action_space=False
        # ),
        # Tennis(
        #     folder="./tennis/tenis_x",
        #     full_action_space=False
        # ),
        # Tennis(
        #     folder="./tennis/tenis_pong_y",
        #     full_action_space=False
        # ),
        # Tennis(
        #     folder="./tennis/tenis_pong_y_full_action_space",
        #     full_action_space=True
        # ),
    ]

    for game in games:
        run(game)