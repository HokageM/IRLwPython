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
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x) # torch.nn.functional.softmax(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.theta_layer = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x_ = nn.functional.relu(self.fc1(x))
        x_ = nn.functional.relu(self.fc2(x_))
        theta_ = self.theta_layer(x_)
        return self.fc3(x_) + torch.matmul(theta_, x)


class MaxEntropyDeepIRL:
    def __init__(self, target, state_dim, action_dim, learning_rate=0.001, gamma=0.99, num_epochs=1000):
        self.target = target
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        # self.theta = torch.rand(state_dim + 1, requires_grad=True)
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.actor_network = ActorNetwork(state_dim, action_dim, 100)
        self.critic_network = CriticNetwork(state_dim + 1, 100)
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=learning_rate)

    def get_reward(self, state, action):
        state_action = list(state) + list([action])
        state_action = torch.Tensor(state_action)
        return self.critic_network(state_action)

    def expert_feature_expectations(self, demonstrations):
        feature_expectations = torch.zeros(self.state_dim)

        for demonstration in demonstrations:
            for state, _, _ in demonstration:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                feature_expectations += state_tensor.squeeze()

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def maxent_irl(self, expert, learner):
        # Update critic network

        self.optimizer_critic.zero_grad()

        # Loss function for critic network
        loss_critic = torch.nn.functional.mse_loss(learner, expert)
        loss_critic.backward()

        self.optimizer_critic.step()

    def update_q_network(self, state_array, action, reward, next_state):
        self.optimizer_actor.zero_grad()

        state_tensor = torch.tensor(state_array, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.actor_network(state_tensor)
        # q_1 = self.actor_network(state_tensor)[action]
        # q_2 = reward + self.gamma * max(self.actor_network(next_state_tensor))
        next_q_values = reward + self.gamma * self.actor_network(next_state_tensor)

        loss_actor = nn.functional.mse_loss(q_values, next_q_values)
        loss_actor.backward()
        self.optimizer_actor.step()

    def get_demonstrations(self):
        env_low = self.target.observation_space.low
        env_high = self.target.observation_space.high
        env_distance = (env_high - env_low) / 20  # self.one_feature

        raw_demo = np.load(file="expert_demo/expert_demo.npy")
        demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))
        for x in range(len(raw_demo)):
            for y in range(len(raw_demo[0])):
                position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
                velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
                state_idx = position_idx + velocity_idx * 20  # self.one_feature

                demonstrations[x][y][0] = state_idx
                demonstrations[x][y][1] = raw_demo[x][y][2]

        return demonstrations

    def train(self):
        demonstrations = self.get_demonstrations()
        expert = self.expert_feature_expectations(demonstrations)

        learner_feature_expectations = torch.zeros(self.state_dim, requires_grad=True)  # Add requires_grad=True
        episodes, scores = [], []

        for episode in range(self.num_epochs):
            state, info = self.target.reset()
            score = 0

            if (episode != 0 and episode == 10) or (episode > 10 and episode % 5 == 0):
                learner = learner_feature_expectations / episode
                self.maxent_irl(expert, learner)

            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32)

                q_state = self.actor_network(state_tensor)
                action = torch.argmax(q_state).item()
                next_state, reward, done, _, _ = self.target.step(action)

                irl_reward = self.get_reward(state, action)
                self.update_q_network(state, action, irl_reward, next_state)

                print("Q Actor Network", state, q_state)
                print("Reward", reward, "IRL Reward", irl_reward)

                learner_feature_expectations = learner_feature_expectations + state_tensor.squeeze()

                print(expert)
                print(learner_feature_expectations)

                score += reward
                state = next_state
                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            if episode % 1 == 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                plt.savefig("./learning_curves/maxent_30000_network.png")

        torch.save(self.q_network.state_dict(), "./results/maxent_30000_q_network.pth")

    def test(self):
        episodes, scores = [], []

        for episode in range(10):
            state = self.target.reset()
            score = 0

            while True:
                self.target.render()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                action = torch.argmax(self.q_network(state_tensor)).item()
                next_state, reward, done, _, _ = self.target.step(action)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    plt.plot(episodes, scores, 'b')
                    plt.savefig("./learning_curves/maxent_test_30000_network.png")
                    break

            if episode % 1 == 0:
                print('{} episode score is {:.2f}'.format(episode, score))
