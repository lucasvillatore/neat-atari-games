import argparse
from src.game import GameGym
from src.space_invaders.neat     import NeatTraining


def get_parameters():
    parser = argparse.ArgumentParser(description='Training to play space invaders')
    parser.add_argument('--generations', type=int, help='Number of generations', required=True)
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = get_parameters()
    
    space_invaders = GameGym(game='SpaceInvaders-v4', is_training=True, player=NeatTraining(args))
    space_invaders.run()
    