from src.game import GameGym
from random import randrange

class RandomPlayer:
    
    def get_action(self):
        return randrange(6)
    
if __name__ == '__main__':
    space_invaders = GameGym(game='SpaceInvaders-v4', is_training=False, player=RandomPlayer())
    space_invaders.run()