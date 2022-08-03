import gym

class   GameGym:
    def __init__(self, game=None, is_training=True, player=None):
        self.game = game
        self.is_training = is_training
        self.render_mode = None if is_training else 'human'
        self.player = player
        
        self.game_state = {
            'actual_state': None,
            'reward_for_action': None,
            'is_finished': False,
            'information': None
        }
    
    def run(self):
        env = gym.make(
                self.game, 
                render_mode=self.render_mode
        )
        
        env.reset()
        
        while not self.game_is_finished():
            
            self.make_new_action(env)

        env.close()
    
    def game_is_finished(self):
        if self.game_state['is_finished']:
            return True
        
        return False
    
    def make_new_action(self, env):
        actual_state_of_game, reward_for_action, game_is_finished, game_information = env.step(self.player.get_action())
        
        self.game_state['actual_state'] = actual_state_of_game
        self.game_state['reward_for_action'] = reward_for_action
        self.game_state['is_finished'] = game_is_finished
        self.game_state['information'] = game_information