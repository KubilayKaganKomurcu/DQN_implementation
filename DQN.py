import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from gym import wrappers


class dense_network:
    def __init__(self, input_size, output_size, activation_function=tf.nn.relu):
        self.weight = tf.Variable(tf.random_normal(shape=(input_size, output_size)))
        self.bias = tf.Variable(np.zeros(output_size).astype(np.float32))
        self.activation_function = activation_function
        self.parameters = [self.weight]

    def forward(self, X):
        net = tf.matmul(X, self.weight) + self.bias
        return self.activation_function(net)


class DQN:
    def __init__(self, layer_sizes, input_size, output_size, session,
                 max_experiences=10000, min_experiences=50, gamma=0.9, batch_size=50):
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.experience = {'states': [], 'actions': [], 'rewards': [], 'states_next': [], 'done': []}
        self.N = 0
        self.session = session

        self.layers = []
        first_layer = dense_network(input_size, layer_sizes[0])
        self.layers.append(first_layer)
        middle_layer = dense_network(layer_sizes[0], layer_sizes[1])
        self.layers.append(middle_layer)
        final_layer = dense_network(layer_sizes[1], output_size)
        self.layers.append(final_layer)

        self.parameters = []
        for layer in self.layers:
            self.parameters += layer.parameters

    def initialize_observation(self, env):
        observation = env.reset()
        observation_size = observation.shape[0]
        return observation, observation_size

    def initialize_network(self, input_size):
        states = tf.placeholder(tf.float32, shape=(None, input_size), name='state')
        states_next = tf.placeholder(tf.float32, shape=(None, input_size), name='state_next')
        actions = tf.placeholder(tf.int32, shape=(None), name='action')
        rewards = tf.placeholder(tf.float32, shape=(None), name='reward')
        # done = tf.placeholder(tf.float32, shape=(None,), name='done')
        targets = tf.placeholder(tf.float32, shape=(None), name='target')

        buffer = [states, actions, rewards, states_next, targets]

        return buffer

    def add_experience(self, states, actions, rewards, states_next, done):
        if len(self.experience['states']) >= self.max_experiences:
            self.experience['states'].pop(0)
            self.experience['actions'].pop(0)
            self.experience['rewards'].pop(0)
            self.experience['states_next'].pop(0)
            self.experience['done'].pop(0)
        self.experience['states'].append(states)
        self.experience['actions'].append(actions)
        self.experience['rewards'].append(rewards)
        self.experience['states_next'].append(states_next)
        self.experience['done'].append(done)

    def Optimizer(self, buffer):

        states = buffer[0]
        actions = buffer[1]
        targets = buffer[4]

        for layer in self.layers:
            states = layer.forward(states)

        calculated_net = states

        one_hot_vector_actions = tf.one_hot(actions, 2)
        selected_action = tf.reduce_sum(targets * one_hot_vector_actions, reduction_indices=[1])

        # Q_opt = rewards + self.gamma * tf.reduce_max(Q)
        # Q_next_opt = rewards + self.gamma * tf.reduce_max(Q_next)

        loss = tf.reduce_sum(tf.square(calculated_net - selected_action))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

        return calculated_net, optimizer

    def EpsGreedy(self, buffer, calculated_net, observation, env, eps_start=1):

        eps = eps_start / (np.sqrt(self.N + 1))
        prob = np.random.random()

        if prob < eps:
            action = env.action_space.sample()
        else:
            obs = np.atleast_2d(observation)
            action = np.argmax(self.predict(obs, calculated_net, buffer)[0])

        self.N += 1
        return action

    def predict(self, state, calculated_net, buffer):
        actions = buffer[0]
        two_d_state = np.atleast_2d(state)
        return self.session.run(calculated_net, feed_dict={actions: two_d_state})

    def train(self, target, optimizer, buffer):

        if len(self.experience['states']) < self.min_experiences:
            return

        batch_selection = np.random.choice(len(self.experience['states']), size=self.batch_size, replace=False)

        states = [self.experience['states'][i] for i in batch_selection]
        actions = [self.experience['actions'][i] for i in batch_selection]
        rewards = [self.experience['rewards'][i] for i in batch_selection]
        next_states = [self.experience['states_next'][i] for i in batch_selection]
        dones = [self.experience['done'][i] for i in batch_selection]

        next_Q = np.max(target.predict(next_states), axis=1)
        targets = [r + self.gamma * next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        self.session.run(optimizer, feed_dict={
            buffer[0]: states,
            buffer[1]: actions,
            buffer[4]: targets}
                         )

    def copy_parameters(self, other):
        ops = []
        my_parameters = self.parameters
        other_parameters = other.parameters
        for i, k in zip(my_parameters, other_parameters):
            actual = self.session.run(k)
            op = i.assign(actual)
            ops.append(op)

        self.session.run(ops)


def play_one(env, model, train_model, eps, copy_period, optimizer, buffer, calculated_net):
    observation = env.reset()
    done = False
    total_reward = 0
    i = 0
    while not done and i < 2000:
        action = model.EpsGreedy(buffer, calculated_net, observation, env, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        total_reward += reward

        model.add_experience(prev_observation, action, reward, observation, done)
        model.train(train_model, optimizer, buffer)

        i += 1

        if i % copy_period == 0:
            train_model.copy_parameters(model)

    return total_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    copy_period = 50

    input_size = len(env.observation_space.sample())
    output_size = env.action_space.n
    layer_sizes = [200, 200]

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()

    model = DQN(layer_sizes, input_size, output_size, session)
    observation_model, observation_size_model = model.initialize_observation(env)

    buffer = model.initialize_network(observation_size_model)
    calculated_net, optimizer = model.Optimizer(buffer)

    train_model = DQN(layer_sizes, input_size, output_size, session)
    observation_train_model, observation_size_train_model = train_model.initialize_observation(env)





    train_model = DQN(layer_sizes, input_size, output_size, session)

    N = 500
    total_rewards = np.empty(N)
    for n in range(N):
        eps_start = 1.0
        totalreward = play_one(env, model, train_model, eps_start, copy_period, optimizer, buffer, calculated_net)
        total_rewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "avg reward (last 100):",
                  total_rewards[max(0, n - 100):(n + 1)].mean())

    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
    print("total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    env = wrappers.Monitor(env, 'shadow_cartpole_dir')

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



    def One_Round(self):
        action = self.env.action_space.sample()
        observation, reward, done, info = self.env.step(action)
        Q = reward + self.gamma *
        
        
        
        
        
    DQN = DQN()
    env = DQN.initialize_environment()
    observation, observation_size = DQN.initialize_observation(env)
    buffer = DQN.initialize_network(observation_size)
    calculated_net, optimizer = DQN.Optimizer(buffer)
'''
