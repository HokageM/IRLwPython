import math
import os

import gym
import numpy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_output, hidden_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(5, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc4(x)
        x = self.fc3(x)
        return x


class DiscreteMaxEntropyDeepIRL:
    def __init__(self, target, state_dim, action_dim, feature_matrix=None, theta=None, learning_rate=0.01, gamma=0.9,
                 num_epochs=1000):
        self.feat_matrix = feature_matrix
        self.one_feature = 20

        self.target = target
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.actor_network = ActorNetwork(5, 1, 100)
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=0.01)

        self.gamma = gamma
        self.num_epochs = num_epochs

        self.theta = theta

    def get_reward(self, state, action):
        state_idx = self.target.state_to_idx(state)
        irl_rewards = self.feat_matrix.dot(self.theta).reshape((400,))
        return irl_rewards[int(state_idx)]

    def expert_feature_expectations(self, demonstrations):
        feature_expectations = np.zeros(self.feat_matrix.shape[0])

        for demonstration in demonstrations:
            for state_idx, _, _ in demonstration:
                feature_expectations += self.feat_matrix[int(state_idx)]

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def maxent_irl(self, expert, learner):
        gradient = expert - learner
        self.theta += self.learning_rate * gradient.detach().numpy()

        # Clip theta
        for j in range(len(self.theta)):
            if self.theta[j] > 0:  # log values
                self.theta[j] = 0

    def update_q_network(self, state_array, action, reward, next_state):
        self.optimizer_actor.zero_grad()

        state_action = list(state_array)
        if action == 0:
            state_action += [1.0, 0.0, 0.0]
        if action == 1:
            state_action += [0.0, 1.0, 0.0]
        if action == 2:
            state_action += [0.0, 0.0, 1.0]

        state_action_tensor = torch.tensor(state_action, dtype=torch.float32)

        state_0 = list(next_state)
        state_0 += [1.0, 0.0, 0.0]
        state_1 = list(next_state)
        state_1 += [0.0, 1.0, 0.0]
        state_2 = list(next_state)
        state_2 += [0.0, 0.0, 1.0]

        state_0_tensor = torch.tensor(state_0, dtype=torch.float32)
        state_1_tensor = torch.tensor(state_1, dtype=torch.float32)
        state_2_tensor = torch.tensor(state_2, dtype=torch.float32)

        next_q_state = torch.tensor([self.actor_network(state_0_tensor),
                                     self.actor_network(state_1_tensor),
                                     self.actor_network(state_2_tensor)])
        soft = nn.Softmax()
        temperatur = 1.5
        next_state_abs = torch.Tensor(soft(next_q_state / temperatur))
        print("Next Q State Softmax", next_state_abs)

        q_1 = self.actor_network(state_action_tensor)
        q_2 = torch.tensor([reward + self.gamma * torch.max(next_state_abs) + q_1])

        loss = torch.nn.functional.mse_loss(q_2, q_1)
        print("Loss", loss)
        loss.backward()
        self.optimizer_actor.step()

    def train(self):
        demonstrations = self.target.get_demonstrations()
        expert = self.expert_feature_expectations(demonstrations)
        expert = torch.Tensor(expert)
        learner_feature_expectations = torch.zeros(400)
        episodes, scores = [], []

        for episode in range(self.num_epochs):
            state, info = self.target.env_reset()
            score = 0

            iteration = 0
            while True:
                iteration += 1
                if iteration > 300:
                    scores.append(-1000)
                    episodes.append(episode)
                    break
                state_idx = self.target.state_to_idx(state)
                state_discrete = self.target.discretize_state(state)

                state_0 = list(state_discrete)
                state_0 += [1.0, 0.0, 0.0]
                state_1 = list(state_discrete)
                state_1 += [0.0, 1.0, 0.0]
                state_2 = list(state_discrete)
                state_2 += [0.0, 0.0, 1.0]

                state_0_tensor = torch.tensor(state_0, dtype=torch.float32)
                state_1_tensor = torch.tensor(state_1, dtype=torch.float32)
                state_2_tensor = torch.tensor(state_2, dtype=torch.float32)

                q_state = torch.tensor([self.actor_network(state_0_tensor),
                                        self.actor_network(state_1_tensor),
                                        self.actor_network(state_2_tensor)])

                soft = nn.Softmax()
                action_probs = torch.Tensor(soft(q_state))

                action = torch.multinomial(action_probs, 1).item()
                next_state, reward, done, _, _ = self.target.env_step(action)

                next_state_discrete = self.target.discretize_state(next_state)

                # Actor update
                irl_reward = self.get_reward(state, action)
                os.system('clear' if os.name == 'posix' else 'cls')
                print("State", state_discrete)
                print("Action", action)
                print("Reward", reward, "IRL_reward", irl_reward)
                print("Q State", q_state)
                # print("Q State Softmax", action_probs)
                self.update_q_network(state, action, irl_reward, next_state_discrete)

                learner_feature_expectations = learner_feature_expectations + torch.Tensor(
                    self.feat_matrix[int(state_idx)])

                score += reward
                state = next_state
                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            # Critic update
            if episode > 4:
                learner = learner_feature_expectations / episode
            else:
                learner = learner_feature_expectations
            self.maxent_irl(expert, learner)

            if episode % 1 == 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                # plt.savefig("./learning_curves/discretemaxentdeep_30000.png")

        torch.save(self.actor_network.state_dict(), "./results/discretemaxentdeep_30000_actor.pth")
