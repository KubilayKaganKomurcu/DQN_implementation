import gym
import numpy as np
import matplotlib.pyplot as plt
import torch


class DQN:
    def __innit__(self, action, state, next_state, gamma=0.1):
        self.gamma = gamma
        self.N = 0

    def EpsGreedy(self, eps_start=1):
        self.N += 1
        eps = eps_start/np.log2(self.N+1)
        prob = np.random.random()

        if prob < eps:
            # random choice
        else:
            # optimal choice

    def DeepQNetwork(self, state, next_state):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        Quality = reward + self.gamma * # (approximation of max future reward) #

env = gym.make('CartPole-v0')

for i_episode in range(20):

    observation = env.reset()

    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
