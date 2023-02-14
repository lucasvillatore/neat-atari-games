
from trainer import run
from pong.interface import Pong, PongXY, PongY
from breakout.interface import Breakout, BreakoutX, BreakoutXY

if __name__ == '__main__':

    games = [
        # Pong(
        #     folder='./pong/pong_x_y', 
        #     full_action_space=False,
        #     tmp=PongXY(),
        # ),
        # Pong(
        #     folder='./pong/pong_x_y_full_action_space', 
        #     full_action_space=True,
        #     tmp=PongXY(),
        # ),
        # Pong(
        #     folder='./pong/pong_y', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_y_full_action_space', 
        #     full_action_space=True,
        #     tmp=PongY(),
        # ),

        Breakout(
            folder='./breakout/breakout_x_full_action_space',
            full_action_space=True,
            tmp=BreakoutX(),
        ),

        Breakout(
            folder='./breakout/breakout_x_y_full_action_space',
            full_action_space=True,
            tmp=BreakoutXY(),
        ),

        # Breakout(
        #     folder='./breakout/breakout_pong_y_full_action_space',
        #     full_action_space=True,
        #     tmp=BreakoutX(),
        #     checkpoint='./pong/pong_y_full_action_space/checkpoints/neat-checkpoint-49'
        # ),

        # Breakout(
        #     folder='./breakout/breakout_pong_x_y_full_action_space',
        #     full_action_space=True,
        #     tmp=BreakoutXY(),
        #     checkpoint='./pong/pong_x_y_full_action_space/checkpoints/neat-checkpoint-49'
        # ),
    ]

    for game in games:
        run(game)