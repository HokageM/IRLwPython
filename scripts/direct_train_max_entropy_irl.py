#
# This file is a refactored implementation of the Maximum Entropy IRL from:
# https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent
# It is a class type implementation restructured for our use case.
#

import gym
import numpy as np
import matplotlib.pyplot as plt


class MaxEntropyIRL:
    def __init__(self, env, feature_matrix, one_feature, q_table, q_learning_rate, gamma, n_states, theta):
        self.env = env
        self.feature_matrix = feature_matrix
        self.one_feature = one_feature
        self.q_table = q_table
        self.q_learning_rate = q_learning_rate
        self.theta = theta
        self.gamma = gamma
        self.n_states = n_states

    def get_feature_matrix(self):
        """
        Returns the feature matrix.
        :return:
        """
        return self.feature_matrix

    def get_reward(self, n_states, state_idx):
        """
        Returns the achieved reward.
        :param n_states:
        :param state_idx:
        :return:
        """
        irl_rewards = self.feature_matrix.dot(self.theta).reshape((n_states,))
        return irl_rewards[state_idx]

    def get_demonstrations(self):
        """
        Parses the demonstrations and returns the demonstrations.
        :param one_feature:
        :return:
        """
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / self.one_feature

        raw_demo = np.load(file="src/irlwpython/expert_demo/expert_demo.npy")
        demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))
        for x in range(len(raw_demo)):
            for y in range(len(raw_demo[0])):
                position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
                velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
                state_idx = position_idx + velocity_idx * self.one_feature

                demonstrations[x][y][0] = state_idx
                demonstrations[x][y][1] = raw_demo[x][y][2]

        return demonstrations

    def expert_feature_expectations(self, demonstrations):
        """
        Returns the feature expectations.
        :param demonstrations:
        :return:
        """
        feature_expectations = np.zeros(self.feature_matrix.shape[0])

        for demonstration in demonstrations:
            for state_idx, _, _ in demonstration:
                feature_expectations += self.feature_matrix[int(state_idx)]

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def state_to_idx(self, env, state):
        """
        Converts state (pos, vel) to the integer value using the mountain car environment.
        :param state:
        :return:
        """
        """ """
        env_low = env.observation_space.low
        env_high = env.observation_space.high
        env_distance = (env_high - env_low) / self.one_feature
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
        state_idx = position_idx + velocity_idx * self.one_feature
        return state_idx

    def maxent_irl(self, expert, learner, learning_rate):
        """
        Max Entropy Learning step.
        :param expert:
        :param learner:
        :param learning_rate:
        :return:
        """
        gradient = expert - learner
        self.theta += learning_rate * gradient

        # Clip theta
        for j in range(len(self.theta)):
            if self.theta[j] > 0:
                self.theta[j] = 0

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q table for a specified state and action.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        """
        q_1 = self.q_table[state][action]
        q_2 = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.q_learning_rate * (q_2 - q_1)


# Training Loop
def train(agent, env, theta_learning_rate, episode_count=30000):
    demonstrations = agent.target.get_demonstrations()
    expert = agent.expert_feature_expectations(demonstrations)
    learner_feature_expectations = np.zeros(agent.n_states)

    episodes, scores = [], []
    for episode in range(episode_count):
        state, info = env.reset()
        score = 0

        # Mini-Batches:
        if (episode + 1) % 10 == 0:
            # calculate density
            learner = learner_feature_expectations / episode
            learner_feature_expectations = np.zeros(agent.n_states)

            agent.maxent_irl(expert, learner, theta_learning_rate)

        state = state
        while True:
            state_idx = agent.state_to_idx(env, state)
            action = np.argmax(agent.q_table[state_idx])

            # Run one timestep of the environment's dynamics.
            next_state, reward, done, _, _ = env.step(action)

            # Get pseudo-reward and update q table
            irl_reward = agent.get_reward(agent.n_states, state_idx)
            next_state_idx = agent.state_to_idx(env, next_state)
            agent.update_q_table(state_idx, action, irl_reward, next_state_idx)

            # State counting for densitiy
            learner_feature_expectations += agent.feature_matrix[int(state_idx)]

            score += reward
            state = next_state
            if done:
                scores.append(score)
                episodes.append(episode)
                break

        if (episode + 1) % 1000 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            save_plot_as_png(episodes, scores,
                             f"src/irlwpython/learning_curves/maxent_{episode_count}_{episode}_qtable.png")
            save_heatmap_as_png(learner.reshape((20, 20)),
                                f"src/irlwpython/heatmap/learner_{episode}_flat.png")
            save_heatmap_as_png(agent.theta.reshape((20, 20)),
                                f"src/irlwpython/heatmap/theta_{episode}_flat.png")

            np.save(f"src/irlwpython/results/maxent_{episode}_qtable", arr=agent.q_table)


def save_heatmap_as_png(data, output_path, title=None, xlabel="Position", ylabel="Velocity"):
    """
    Create a heatmap from a numpy array and save it as a PNG file.
    :param data: 2D numpy array containing the heatmap data.
    :param output_path: Output path for saving the PNG file.
    :param xlabel: Label for the x-axis (optional).
    :param ylabel: Label for the y-axis (optional).
    :param title: Title for the plot (optional).
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar(im)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.savefig(output_path, format='png')
    plt.close(fig)


def save_plot_as_png(x, y, output_path, title=None, xlabel="Episodes", ylabel="Scores"):
    """
    Create a line plot from x and y data and save it as a PNG file.
    :param x: 1D numpy array or list representing the x-axis values.
    :param y: 1D numpy array or list representing the y-axis values.
    :param output_path: Output path for saving the plot as a PNG file.
    :param xlabel: Label for the x-axis (optional).
    :param ylabel: Label for the y-axis (optional).
    :param title: Title for the plot (optional).
    """
    fig, ax = plt.subplots()
    ax.plot(x, y)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.savefig(output_path, format='png')
    plt.close(fig)


# Main function
if __name__ == "__main__":
    n_states = 400  # position - 20, velocity - 20 -> 20*20
    n_actions = 3  # Accelerate to the left: 0, Don’t accelerate: 1, Accelerate to the right: 2
    state_dim = 2  # Velocity and position
    one_feature = 20
    feature_matrix = np.eye(n_states)

    gamma = 0.99
    q_learning_rate = 0.03

    # Theta works as Rewards
    theta_learning_rate = 0.001
    theta = -(np.random.uniform(size=(n_states,)))

    env = gym.make('MountainCar-v0')

    q_table = np.zeros((n_states, n_actions))
    agent = MaxEntropyIRL(env, feature_matrix, one_feature, q_table, q_learning_rate, gamma, n_states, theta)

    train(agent, env, theta_learning_rate)
