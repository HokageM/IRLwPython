#
# This file is a refactored implementation of the Maximum Entropy IRL from:
# https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent
# It is a class type implementation restructured for our use case.
#

import numpy as np
import matplotlib.pyplot as plt


class MaxEntropyIRL:
    def __init__(self, target, feature_matrix, one_feature, q_table, q_learning_rate, gamma, n_states, theta):
        self.target = target
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

    def train(self, theta_learning_rate, episode_count=30000):
        """
        Trains a model.
        :param theta_learning_rate:
        :return:
        """
        demonstrations = self.target.get_demonstrations()
        expert = self.expert_feature_expectations(demonstrations)
        learner_feature_expectations = np.zeros(self.n_states)

        episodes, scores = [], []
        for episode in range(episode_count):
            state, info = self.target.env_reset()
            score = 0

            # Mini-Batches:
            if (episode != 0 and episode == 10000) or (
                    episode > 10000 and episode % 5000 == 0):  # % 100, and reset learner
                # calculate density
                learner = learner_feature_expectations / episode
                self.maxent_irl(expert, learner, theta_learning_rate)

            state = state
            while True:
                state_idx = self.target.state_to_idx(state)
                action = np.argmax(self.q_table[state_idx])

                # Run one timestep of the environment's dynamics.
                next_state, reward, done, _, _ = self.target.env_step(action)

                # Get pseudo-reward and update q table
                irl_reward = self.get_reward(self.n_states, state_idx)
                next_state_idx = self.target.state_to_idx(next_state)
                self.update_q_table(state_idx, action, irl_reward, next_state_idx)

                # State counting for densitiy
                learner_feature_expectations += self.feature_matrix[int(state_idx)]

                score += reward
                state = next_state
                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            if episode % 1000 == 0 and episode != 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                plt.savefig(f"src/irlwpython/learning_curves/maxent_{episode}_qtable.png")
                np.save(f"src/irlwpython/results/maxent_{episode}_qtable", arr=self.q_table)
                learner = learner_feature_expectations / episode
                plt.imshow(learner.reshape((20, 20)), cmap='viridis', interpolation='nearest')
                plt.savefig(f"src/irlwpython/heatmap/learner_{episode}_qtable.png")
                plt.imshow(self.theta.reshape((20, 20)), cmap='viridis', interpolation='nearest')
                plt.savefig(f"src/irlwpython/heatmap/theta_{episode}_qtable.png")
                plt.imshow(self.feature_matrix.dot(self.theta).reshape((20, 20)), cmap='viridis',
                           interpolation='nearest')
                plt.savefig(f"src/irlwpython/heatmap/rewards_{episode}_qtable.png")

    def test(self):
        """
        Tests the previous trained model
        :return:
        """
        episodes, scores = [], []

        for episode in range(10):
            state = self.target.env_reset()
            score = 0

            state = state[0]
            while True:
                self.target.env_render()
                state_idx = self.target.state_to_idx(state)
                action = np.argmax(self.q_table[state_idx])
                next_state, reward, done, _, _ = self.target.env_step(action)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    plt.plot(episodes, scores, 'b')
                    plt.savefig("src/irlwpython/learning_curves/maxent_test_30000_maxentropy.png")
                    break

            if episode % 1 == 0:
                print('{} episode score is {:.2f}'.format(episode, score))
