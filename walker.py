import gym
import time

ENVIRONMENT = 'BipedalWalker-v3'


def main():
    environment = get_env_gym()

    observation_space = environment.observation_space
    action_space = environment.action_space

    print("The observation space: {}".format(observation_space))
    print("The action space: {}".format(action_space))

    initial_observation = environment.reset()
    print("The initial observation is {}".format(initial_observation))

    number_of_steps = 1500
    
    training(environment, number_of_steps)

    environment.close()

def get_env_gym():
    return gym.make(ENVIRONMENT)

def training(environment, number_of_steps):
    for step in range(number_of_steps):
        random_action = environment.action_space.sample()
        
        new_observation, reward, done, info = environment.step(random_action)
        
        print("The new observation is {}".format(new_observation))
        
        environment.render(mode = "human")

        if done:
            break

        time.sleep(0.01)

if __name__ == '__main__':
    main()