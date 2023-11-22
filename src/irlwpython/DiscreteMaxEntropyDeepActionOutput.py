import math
import os

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_output, hidden_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(QNetwork, self).__init__()

        # Define the layers of your neural network
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),  # You can use other activation functions here
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        # Define the forward pass of your neural network
        return self.layers(x)

class DiscreteMaxEntropyDeepIRL:
    def __init__(self, target, state_dim, action_dim, feature_matrix=None, theta=None, learning_rate=0.05, gamma=0.99,
                 num_epochs=1000):
        self.feat_matrix = feature_matrix
        self.one_feature = 20

        self.target = target
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        #self.actor_network = ActorNetwork(2, 3, 100)
        self.actor_network = QNetwork(2, 3, 100)
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

        state_tensor = torch.tensor(state_array, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        #soft = nn.Softmax()
        #q_state = soft(self.actor_network(state_tensor))
        q_state = self.actor_network(state_tensor)



        #next_q_state = soft(self.actor_network(next_state_tensor))
        next_q_state = self.actor_network(next_state_tensor)
        print("Next QSTATE", next_q_state)

        q_1 = q_state[action]
        q_2 = reward + self.gamma * torch.max(next_q_state) + q_1

        #if action == 0:
        #    loss = torch.nn.functional.mse_loss(torch.tensor([q_2, 0, 0], requires_grad=True), q_state)
        #if action == 1:
        #    loss = torch.nn.functional.mse_loss(torch.tensor([0, q_2, 0], requires_grad=True), q_state)
        #if action == 2:
        #    loss = torch.nn.functional.mse_loss(torch.tensor([0, 0, q_2], requires_grad=True), q_state)

        loss = torch.nn.functional.mse_loss(q_2, q_1)

        loss = loss #* q_1
        print("LOSS", loss)
        #loss = loss.sum()
        #print("LOSS", loss)
        loss.backward()

        self.optimizer_actor.step()

    def train(self):
        demonstrations = self.target.get_demonstrations()
        expert = self.expert_feature_expectations(demonstrations)
        expert = torch.Tensor(expert)
        learner_feature_expectations = torch.zeros(400)
        episodes, scores = [], []

        steps = 0
        for episode in range(20):#self.num_epochs):
            state, info = self.target.env_reset()
            score = 0

            iteration = 0
            while True:
                steps += 1
                iteration += 1
                if iteration > 2000:
                    scores.append(-1000)
                    episodes.append(episode)
                    break
                state_idx = self.target.state_to_idx(state)
                state_discrete = self.target.discretize_state(state)

                state_tensor = torch.tensor(state_discrete, dtype=torch.float32)

                q_state = self.actor_network(state_tensor)

                #action = torch.argmax(q_state).item()
                action = torch.multinomial(q_state, 1).item()
                next_state, reward, done, _, _ = self.target.env_step(action)

                next_state_discrete = self.target.discretize_state(next_state)

                # Actor update
                irl_reward = self.get_reward(state, action)
                #os.system('clear' if os.name == 'posix' else 'cls')
                print(episode)
                print("STATE", state_discrete)
                print("ACTION", action)
                print("Q state", q_state)
                print("Reward", irl_reward)
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
            if episode != 0:
                learner = learner_feature_expectations / episode
            else:
                learner = learner_feature_expectations
            self.maxent_irl(expert, learner)

            if episode == 99:
                print("Expert", expert)
                print("Learner", learner)
                for state_1 in range(20):
                    for action_1 in range(20):
                        print("State: ", state_1, action_1, "Q Values",
                              self.actor_network(torch.tensor([state_1, action_1], dtype=torch.float32)),
                              "Reward: ", self.get_reward(state, action))


            if episode % 1 == 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                # plt.savefig("./learning_curves/discretemaxentdeep_30000.png")

        torch.save(self.actor_network.state_dict(), "./results/discretemaxentdeep_30000_actor.pth")

