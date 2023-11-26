import math
import os
import random

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(128, 128)
        # self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        q_values = self.output_layer(x)
        return q_values


class MaxEntropyDeepIRL:
    def __init__(self, target, state_dim, action_size, feature_matrix=None, one_feature=None, theta=None,
                 learning_rate=0.05, gamma=0.9,
                 num_epochs=1000):
        self.feature_matrix = feature_matrix
        self.one_feature = one_feature

        self.target = target
        self.state_dim = state_dim
        self.action_dim = action_size

        self.q_network = QNetwork(state_dim, action_size)
        self.target_q_network = QNetwork(state_dim, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.num_epochs = num_epochs

        self.theta_learning_rate = 0.05
        self.theta = theta

    def tensor_to_image(self, tensor, name):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        tensor_img = PIL.Image.fromarray(tensor)
        tensor_img.save(f"{name}.png", "PNG")

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(3)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return torch.argmax(q_values).item()

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
        feature_expectations = np.zeros(self.feature_matrix.shape[0])

        for demonstration in demonstrations:
            for state_idx, _, _ in demonstration:
                feature_expectations += self.feature_matrix[int(state_idx)]

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def maxent_irl(self, expert, learner):
        """
        Max Entropy Learning step.
        :param expert:
        :param learner:
        :param learning_rate:
        :return:
        """
        gradient = expert - learner
        self.theta += self.theta_learning_rate * gradient

        # Clip theta
        for j in range(len(self.theta)):
            if self.theta[j] > 0:  # log values
                self.theta[j] = 0

    def update_q_network(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        q_values = self.q_network(state)
        next_q_values = self.target_q_network(next_state)

        target = q_values.clone()
        if not done:
            target[action] = reward + self.gamma * torch.max(next_q_values).item()
        else:
            target[action] = reward

        loss = nn.MSELoss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, n_states, episodes=10000, max_steps=10000,
              epsilon_start=1.0,
              epsilon_decay=0.995, epsilon_min=0.01):
        epsilon = epsilon_start
        episode_arr, scores = [], []

        demonstrations = self.target.get_demonstrations()
        expert = self.expert_feature_expectations(demonstrations)
        learner_feature_expectations = np.zeros(n_states)

        for episode in range(episodes):
            state, info = self.target.env_reset()
            total_reward = 0

            # Mini-Batches:
            if (episode != 0 and episode == 1000) or (episode > 1000 and episode % 500 == 0):
                # calculate density
                learner = learner_feature_expectations / episode
                # Maximum Entropy IRL step
                self.maxent_irl(expert, learner)

            for step in range(max_steps):
                action = self.select_action(state, epsilon)

                next_state, reward, done, _, _ = self.target.env_step(action)
                # Real Reward
                total_reward += reward

                # IRL
                state_idx = self.target.state_to_idx(state)
                irl_reward = self.get_reward(n_states, state_idx)

                self.update_q_network(state, action, irl_reward, next_state, done)
                self.update_target_network()

                # State counting for densitiy
                learner_feature_expectations += self.feature_matrix[int(state_idx)]

                state = next_state
                if done:
                    scores.append(total_reward)
                    episode_arr.append(episode)
                    break

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            if episode % 50 == 0:
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

            if episode == episodes - 1:
                plt.plot(episode_arr, scores, 'b')
                plt.savefig(f"./learning_curves/maxentdeep_{episodes}_qdeep.png")

        torch.save(self.q_network.state_dict(), f"./results/maxentdeep_{episodes}_q_network.pth")
