
from trainer import run
from pong.interface import Pong, PongXY, PongY
from breakout.interface import Breakout, BreakoutX, BreakoutXY

if __name__ == '__main__':

    games = [
        # Pong(
        #     folder='./pong/pong_activation_default_cube', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_activation_default_exp', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_activation_default_relu', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_activation_default_sigmoid', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_activation_default_tanh', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_compatibility_threshold_1.0', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_compatibility_threshold_2.0', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_compatibility_threshold_3.0', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_compatibility_threshold_4.0', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        # Pong(
        #     folder='./pong/pong_compatibility_threshold_6.0', 
        #     full_action_space=False,
        #     tmp=PongY(),
        # ),
        Pong(
            folder='./pong/pong_initial_connection_full_direct', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_initial_connection_unconnected', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_num_hidden_0', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_num_hidden_3', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_num_hidden_6', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_num_hidden_9', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_pop_size_50', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_pop_size_150', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_pop_size_300', 
            full_action_space=False,
            tmp=PongY(),
        ),
        Pong(
            folder='./pong/pong_pop_size_600', 
            full_action_space=False,
            tmp=PongY(),
        ),
    ]

    for game in games:
        run(game)