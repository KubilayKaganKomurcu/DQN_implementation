import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

class DQN:
    def __innit__(self, env, states, actions, rewards, states_next, gamma=0.5):
        super().__innit__()
        self.gamma = gamma
       # self.unstoppable_gamma = gamma
        self.env = env
        self.states = states
        self.states_next = states_next
        self.actions = actions
        self.rewards = rewards
        self.N = 0


    def initialize_environment(self):
        env = gym.make('CartPole-v0')
        return env


    def initialize_observation(self):
        observation = self.env.reset()
        observation_size = observation.shape[0]
        return observation, observation_size


    def initialize_network(self, observation_size):
        batch_size = 50

        states = tf.placeholder(tf.float32, shape=(batch_size, observation_size), name='state')
        states_next = tf.placeholder(tf.float32, shape=(batch_size, observation_size), name='state_next')
        actions = tf.placeholder(tf.int32, shape=(batch_size,), name='action')
        rewards = tf.placeholder(tf.float32, shape=(batch_size,), name='reward')
        #done = tf.placeholder(tf.float32, shape=(batch_size,), name='done')

        buffer = [states, actions, rewards, states_next]

        return buffer

    def CartPoleNetwork(self, actions, rewards):
        Q = tf.layers.dense(self.states, [64,64,2], activation= tf.nn.relu)
        Q_next = tf.layers.dense(self.states_next, [64,64,2], activation= tf.nn.relu)
        one_hot_vector_actions = tf.one_hot(actions, 2)
        pred = tf.reduce_sum(Q * one_hot_vector_actions) #TODO how does this create predictions  ?



        Q_opt = rewards + self.gamma *  tf.reduce_max(Q)        #TODO terminal states shouldn't get gamma terms, fix this.
        Q_next_opt = rewards + self.gamma *  tf.reduce_max(Q_next)


        loss =tf.reduce_mean(tf.square(pred - Q_next_opt))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


    def EpsGreedy(self, eps_start=1):

        eps = eps_start/(self.N+1)
        prob = np.random.random()

        if prob < eps:
            action = self.env.action_space.sample() # random choice
        else:
            action = argmax()                       #TODO optimal choice

        self.N += 1
        return action

    def DeepQ(self, ):

        action = self.EpsGreedy()
        observation, reward, done, info = self.env.step(action)
        Q = reward + self.gamma *  #TODO (approximation of max future reward) #


'''
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
'''

