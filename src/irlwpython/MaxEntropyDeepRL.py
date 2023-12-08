import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn as nn

from irlwpython.OutputHandler import OutputHandler


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, output_size)

        self.output_hand = OutputHandler()

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        q_values = self.output_layer(x)
        return q_values


class MaxEntropyDeepRL:
    def __init__(self, target, state_dim, action_size, feature_matrix, one_feature, learning_rate=0.001, gamma=0.99):
        self.feature_matrix = feature_matrix
        self.one_feature = one_feature

        self.target = target

        self.q_network = QNetwork(state_dim, action_size)
        self.target_q_network = QNetwork(state_dim, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.gamma = gamma

        self.output_hand = OutputHandler()

    def select_action(self, state, epsilon):
        """
        Selects an action based on the q values from the network with epsilon greedy.
        :param state:
        :param epsilon:
        :return:
        """
        if np.random.rand() < epsilon:
            return np.random.choice(3)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return torch.argmax(q_values).item()

    def update_q_network(self, state, action, reward, next_state, done):
        """
        Updates the q network based on the reward
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
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
        """
        Updates the target network.
        :return:
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, n_states, episodes=30000, max_steps=200,
              epsilon_start=1.0,
              epsilon_decay=0.995, epsilon_min=0.01):
        """
        Trains the network using the maximum entropy deep reinforcement algorithm.
        :param n_states:
        :param episodes: Count of training episodes
        :param max_steps: Max steps per episode
        :param epsilon_start:
        :param epsilon_decay:
        :param epsilon_min:
        :return:
        """
        learner_feature_expectations = np.zeros(n_states)

        epsilon = epsilon_start
        episode_arr, scores = [], []

        best_reward = -math.inf
        for episode in range(episodes):
            state, info = self.target.env_reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.select_action(state, epsilon)

                next_state, reward, done, _, _ = self.target.env_step(action)
                total_reward += reward

                self.update_q_network(state, action, reward, next_state, done)
                self.update_target_network()

                # State counting for densitiy
                state_idx = self.target.state_to_idx(state)
                learner_feature_expectations += self.feature_matrix[int(state_idx)]

                state = next_state
                if done:
                    break

            # Keep track of best performing network
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.q_network.state_dict(),
                           f"../results/maxentropydeep_{episode}_best_network_w_{total_reward}_RL.pth")

            if (episode + 1) % 10 == 0:
                # calculate density
                learner = learner_feature_expectations / episode
                learner_feature_expectations = np.zeros(n_states)

            scores.append(total_reward)
            episode_arr.append(episode)
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

            if (episode + 1) % 1000 == 0:
                score_avg = np.mean(scores)
                print('{} episode average score is {:.2f}'.format(episode, score_avg))
                self.output_hand.save_plot_as_png(episode_arr, scores,
                                              f"../learning_curves/maxent_{episodes}_{episode}_qnetwork_RL.png")
                self.output_hand.save_heatmap_as_png(learner.reshape((20, 20)), f"../heatmap/learner_{episode}_deep_RL.png")
                self.output_hand.save_heatmap_as_png(self.theta.reshape((20, 20)),
                                                 f"../heatmap/theta_{episode}_deep_RL.png")

                torch.save(self.q_network.state_dict(), f"../results/maxent_{episodes}_{episode}_network_main.pth")

            if episode == episodes - 1:
                self.output_hand.save_plot_as_png(episode_arr, scores,
                                              f"../learning_curves/maxentdeep_{episodes}_qdeep_RL.png")

        torch.save(self.q_network.state_dict(), f"src/irlwpython/results/maxentdeep_{episodes}_q_network_RL.pth")

    def test(self, model_path, epsilon=0.01, repeats=100):
        """
        Tests the previous trained model.
        :return:
        """
        self.q_network.load_state_dict(torch.load(model_path))
        episodes, scores = [], []

        for episode in range(repeats):
            state, info = self.target.env_reset()
            score = 0

            while True:
                self.target.env_render()
                action = self.select_action(state, epsilon)
                next_state, reward, done, _, _ = self.target.env_step(action)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            if episode % 1 == 0:
                print('{} episode score is {:.2f}'.format(episode, score))

        self.output_hand.save_plot_as_png(episodes, scores,
                                      "src/irlwpython/learning_curves"
                                      "/test_maxentropydeep_best_model_RL_results.png")
